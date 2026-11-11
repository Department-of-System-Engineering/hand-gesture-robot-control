/*
  ESP32 + ADXL345/PmodACL + L3G4200D/PmodGYRO + pot
  TDK pick-and-place full-system firmware V5.

  PC-oldali adatcsomag:
    IMUP,ms,qw,qx,qy,qz,roll,pitch,yaw,ax,ay,az,gx,gy,gz,pot_raw

  Bekotes a projektben hasznalt kiosztassal:
    SCK  GPIO18
    MISO GPIO19
    MOSI GPIO23
    ADXL CS GPIO5
    GYRO CS GPIO17
    POT wiper GPIO34 (pot ends: 3.3V/GND)
    HAPTIC SERVO signal GPIO25  [TDK pick-place-ben NEM hajtjuk]
    optional button GPIO27 -> GND

  Megjegyzesek:
  - Serial 115200.
  - Bootkor csak gyro bias kalibracio: tartsd mozdulatlanul.
  - 'Z' soros parancs: aktualis IMU orientacio legyen relativ nulla.
    Ez kompatibilis a PC oldali C referencia folyamattal.
  - A haptikus szervo ebben a TDK-meresi firmware-ben szandekosan nincs
    aktivalva; a haptika kulon validacios ag.
*/

#include <Arduino.h>
#include <SPI.h>
#include <math.h>

const int PIN_SCK = 18;
const int PIN_MISO = 19;
const int PIN_MOSI = 23;
const int PIN_ADXL_CS = 5;
const int PIN_GYRO_CS = 17;
const int PIN_POT = 34;
const int PIN_SERVO = 25;
const int PIN_BUTTON = 27;

// A korabbi integralt projectvonalhoz igazodva konzervativ 1 MHz, MODE3.
SPISettings spiCfg(1000000, MSBFIRST, SPI_MODE3);

const float ADXL_G_PER_LSB = 0.0039f;
const float GYRO_DPS_PER_LSB = 0.070f;
const float DEG2RAD = 0.01745329251994329577f;
const float RAD2DEG = 57.295779513082320876f;

// Mahony belso (szenzor abszolut/gravitacios) quaternion.
float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
float iFBx = 0.0f, iFBy = 0.0f, iFBz = 0.0f;
float biasX = 0.0f, biasY = 0.0f, biasZ = 0.0f;

const float twoKp = 1.4f;
const float twoKi = 0.03f;

// Relativ nullahoz tarolt referencia quaternion.
float ref0 = 1.0f, ref1 = 0.0f, ref2 = 0.0f, ref3 = 0.0f;

uint32_t lastUpdateUs = 0;
uint32_t lastPrintUs = 0;
const uint32_t UPDATE_US = 5000;   // 200 Hz fusion
const uint32_t PRINT_US  = 10000;  // 100 Hz serial

void writeReg(int cs, uint8_t reg, uint8_t val) {
  SPI.beginTransaction(spiCfg);
  digitalWrite(cs, LOW);
  SPI.transfer(reg & 0x3F);
  SPI.transfer(val);
  digitalWrite(cs, HIGH);
  SPI.endTransaction();
}

uint8_t readReg(int cs, uint8_t reg) {
  SPI.beginTransaction(spiCfg);
  digitalWrite(cs, LOW);
  SPI.transfer(reg | 0x80);
  uint8_t v = SPI.transfer(0);
  digitalWrite(cs, HIGH);
  SPI.endTransaction();
  return v;
}

void readMulti(int cs, uint8_t reg, uint8_t *b, int n) {
  SPI.beginTransaction(spiCfg);
  digitalWrite(cs, LOW);
  SPI.transfer(reg | 0xC0);  // read + multi/auto-increment
  for (int i = 0; i < n; ++i) b[i] = SPI.transfer(0);
  digitalWrite(cs, HIGH);
  SPI.endTransaction();
}

void initSensors() {
  // ADXL345: ~200 Hz, FULL_RES, +/-16 g, measure.
  writeReg(PIN_ADXL_CS, 0x2C, 0x0B);
  writeReg(PIN_ADXL_CS, 0x31, 0x0B);
  writeReg(PIN_ADXL_CS, 0x2D, 0x08);

  // L3G4200D: normal, XYZ, magas ODR, BDU + +/-2000 dps.
  writeReg(PIN_GYRO_CS, 0x20, 0x8F);
  writeReg(PIN_GYRO_CS, 0x23, 0xA0);
}

void readAccel(float &ax, float &ay, float &az) {
  uint8_t b[6];
  readMulti(PIN_ADXL_CS, 0x32, b, 6);
  int16_t x = (int16_t)(((uint16_t)b[1] << 8) | b[0]);
  int16_t y = (int16_t)(((uint16_t)b[3] << 8) | b[2]);
  int16_t z = (int16_t)(((uint16_t)b[5] << 8) | b[4]);
  ax = x * ADXL_G_PER_LSB;
  ay = y * ADXL_G_PER_LSB;
  az = z * ADXL_G_PER_LSB;
}

void readGyro(float &gx, float &gy, float &gz) {
  uint8_t b[6];
  readMulti(PIN_GYRO_CS, 0x28, b, 6);
  int16_t x = (int16_t)(((uint16_t)b[1] << 8) | b[0]);
  int16_t y = (int16_t)(((uint16_t)b[3] << 8) | b[2]);
  int16_t z = (int16_t)(((uint16_t)b[5] << 8) | b[4]);
  gx = x * GYRO_DPS_PER_LSB;
  gy = y * GYRO_DPS_PER_LSB;
  gz = z * GYRO_DPS_PER_LSB;
}

void mahony(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
  float a2 = ax*ax + ay*ay + az*az;
  if (a2 > 1e-9f) {
    float r = 1.0f / sqrtf(a2);
    ax *= r; ay *= r; az *= r;

    float hvx = q1*q3 - q0*q2;
    float hvy = q0*q1 + q2*q3;
    float hvz = q0*q0 - 0.5f + q3*q3;

    float ex = ay*hvz - az*hvy;
    float ey = az*hvx - ax*hvz;
    float ez = ax*hvy - ay*hvx;

    iFBx += twoKi * ex * dt;
    iFBy += twoKi * ey * dt;
    iFBz += twoKi * ez * dt;

    gx += iFBx + twoKp * ex;
    gy += iFBy + twoKp * ey;
    gz += iFBz + twoKp * ez;
  }

  gx *= 0.5f * dt;
  gy *= 0.5f * dt;
  gz *= 0.5f * dt;

  float a = q0, b = q1, c = q2;
  q0 += -b*gx - c*gy - q3*gz;
  q1 +=  a*gx + c*gz - q3*gy;
  q2 +=  a*gy - b*gz + q3*gx;
  q3 +=  a*gz + b*gy - c*gx;

  float n = sqrtf(q0*q0 + q1*q1 + q2*q2 + q3*q3);
  if (n > 1e-9f) {
    q0/=n; q1/=n; q2/=n; q3/=n;
  }
}

void quatMul(float aw,float ax,float ay,float az,
             float bw,float bx,float by,float bz,
             float &ow,float &ox,float &oy,float &oz) {
  ow = aw*bw - ax*bx - ay*by - az*bz;
  ox = aw*bx + ax*bw + ay*bz - az*by;
  oy = aw*by - ax*bz + ay*bw + az*bx;
  oz = aw*bz + ax*by - ay*bx + az*bw;
}

void getRelativeQuaternion(float &w,float &x,float &y,float &z) {
  // q_rel = inv(q_ref) * q_current. Egysegkvaternionnal az inverz a konjugalt.
  quatMul(ref0, -ref1, -ref2, -ref3, q0, q1, q2, q3, w, x, y, z);
  float n = sqrtf(w*w+x*x+y*y+z*z);
  if (n > 1e-9f) { w/=n; x/=n; y/=n; z/=n; }
  // Azonos forgatasnal stabil elojel a log/PC szamara.
  if (w < 0) { w=-w; x=-x; y=-y; z=-z; }
}

void quatToEulerDeg(float w,float x,float y,float z,float &roll,float &pitch,float &yaw) {
  float sinr_cosp = 2.0f * (w*x + y*z);
  float cosr_cosp = 1.0f - 2.0f * (x*x + y*y);
  roll = atan2f(sinr_cosp, cosr_cosp) * RAD2DEG;

  float sinp = 2.0f * (w*y - z*x);
  if (fabsf(sinp) >= 1.0f) pitch = copysignf(90.0f, sinp);
  else pitch = asinf(sinp) * RAD2DEG;

  float siny_cosp = 2.0f * (w*z + x*y);
  float cosy_cosp = 1.0f - 2.0f * (y*y + z*z);
  yaw = atan2f(siny_cosp, cosy_cosp) * RAD2DEG;
}

void calibrateGyro() {
  Serial.println("# GYRO_CAL_START - tartsd mozdulatlanul");
  const int N = 1000;
  double sx=0, sy=0, sz=0;
  for (int i=0; i<N; ++i) {
    float x,y,z;
    readGyro(x,y,z);
    sx += x; sy += y; sz += z;
    delay(3);
  }
  biasX = sx/N; biasY = sy/N; biasZ = sz/N;
  Serial.printf("# GYRO_BIAS_DPS,%.6f,%.6f,%.6f\n", biasX,biasY,biasZ);
  Serial.println("# GYRO_CAL_DONE");
}

void setRelativeZero() {
  ref0=q0; ref1=q1; ref2=q2; ref3=q3;
  Serial.println("# RELATIVE_ZERO_SET");
}

void handleSerialCommands() {
  static String line;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line.trim();
      if (line.equalsIgnoreCase("Z")) setRelativeZero();
      else if (line.equalsIgnoreCase("PING")) Serial.println("# PONG");
      else if (line.length() > 0) {
        Serial.print("# UNKNOWN_CMD,");
        Serial.println(line);
      }
      line = "";
    } else if (line.length() < 64) {
      line += c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(400);

  pinMode(PIN_ADXL_CS, OUTPUT);
  pinMode(PIN_GYRO_CS, OUTPUT);
  pinMode(PIN_POT, INPUT);
  pinMode(PIN_SERVO, OUTPUT);
  digitalWrite(PIN_SERVO, LOW); // haptika TDK pick-place-ben OFF
  pinMode(PIN_BUTTON, INPUT_PULLUP);

  digitalWrite(PIN_ADXL_CS, HIGH);
  digitalWrite(PIN_GYRO_CS, HIGH);

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI);
  analogReadResolution(12);
  delay(50);

  Serial.printf("# ADXL_ID,0x%02X\n", readReg(PIN_ADXL_CS, 0x00));
  Serial.printf("# GYRO_ID,0x%02X\n", readReg(PIN_GYRO_CS, 0x0F));

  initSensors();
  delay(100);
  calibrateGyro();

  // Elso relatív referencia csak a bootallapot. a PC C-je kuld uj Z-t.
  setRelativeZero();

  lastUpdateUs = micros();
  lastPrintUs = lastUpdateUs;
  Serial.println("# READY");
}

void loop() {
  handleSerialCommands();

  uint32_t now = micros();
  if ((uint32_t)(now - lastUpdateUs) < UPDATE_US) {
    delayMicroseconds(150);
    return;
  }

  float dt = (now - lastUpdateUs) * 1e-6f;
  lastUpdateUs = now;
  if (dt <= 0.0f || dt > 0.05f) return;

  float ax,ay,az,gx,gy,gz;
  readAccel(ax,ay,az);
  readGyro(gx,gy,gz);
  gx -= biasX; gy -= biasY; gz -= biasZ;

  mahony(gx*DEG2RAD, gy*DEG2RAD, gz*DEG2RAD, ax,ay,az,dt);

  if ((uint32_t)(now - lastPrintUs) >= PRINT_US) {
    lastPrintUs = now;

    float w,x,y,z;
    getRelativeQuaternion(w,x,y,z);
    float roll,pitch,yaw;
    quatToEulerDeg(w,x,y,z,roll,pitch,yaw);

    int potRaw = analogRead(PIN_POT);

    Serial.printf(
      "IMUP,%lu,%.7f,%.7f,%.7f,%.7f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d\n",
      millis(),w,x,y,z,roll,pitch,yaw,ax,ay,az,gx,gy,gz,potRaw
    );
  }
}
