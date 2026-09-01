// imu_probe.ino — IMU-only logger for the BNO055 sitting on the torso.
//
// Purpose: decide, with hardware data, whether the device Euler "roll" the kappa-PID
// currently feeds on (imu_roll = euler.y()) still reads lateral lean once the torso is
// pitched ~20-25 deg forward by the hip offset. Sim says a gravity-vector roll
// (atan2 of two gravity components) is pitch-immune by construction; this sketch
// records BOTH plus the raw quaternion, so every decomposition can be redone offline
// from one capture -- no re-testing to try a different convention.
//
// Standalone on purpose: no DynamixelShield, no WiFiNINA. Motors do NOT need power;
// USB from the laptop is enough for the board + IMU. Nothing here commands anything.
//
// Serial: 115200 baud, CSV lines. '#' lines are comments/markers, not data.
//   h      reprint the header
//   space  pause/resume the stream
//   0..9   set the 'pose' column (label the static poses A=1, B=2, ... as you go)
//   z      2-second STATIC average: mean +/- sd of every channel, printed as #AVG
//          (this is the one to use for the bench poses; a single sample is noisy)
//   c      print calibration status only
//
// BNO055 calibration: cal_acc must reach 3 before the gravity vector is trustworthy.
// Move the board slowly through a few orientations and pause a second in each; the
// stream prints the four cal values on every row so you can watch them climb.
//
// Note on style: plain global arrays, no structs. The Arduino IDE auto-inserts function
// prototypes at the top of the file, ahead of any struct definition, so a struct in a
// signature fails to compile ("'Sample' was not declared in this scope").

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#define DEBUG_SERIAL Serial
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

const uint32_t AVG_MS = 2000;       // 'z' static-average window (sampled at 50 Hz)

// Stream rate is adjustable because the terminal is also the log: a 50 Hz stream is
// unreadable while posing. The static poses need no stream at all -- 'z' prints one
// line each -- so the stream starts OFF and only the dynamic pose turns it on.
const uint32_t RATES_MS[] = {200, 100, 50, 20};   // 5, 10, 20, 50 Hz
int      rate_idx  = 1;             // default 10 Hz: readable, still fine for hand motion
bool     streaming = false;         // start quiet
int      pose_id   = 0;
uint32_t t_last    = 0;

// --- latest sample (filled by readSample) ---
float   s_q[4];        // quaternion w,x,y,z
float   s_eul[3];      // device Euler: x=yaw/heading, y=roll, z=pitch  [deg]
float   s_g[3];        // VECTOR_GRAVITY          [m/s^2]
float   s_a[3];        // VECTOR_ACCELEROMETER    [m/s^2, gravity NOT removed]
uint8_t s_cal[4];      // sys, gyro, accel, mag

const char *HDR =
  "ms,pose,qw,qx,qy,qz,eul_yaw,eul_roll,eul_pitch,gx,gy,gz,ax,ay,az,"
  "roll_xy,roll_zy,cal_sys,cal_gyro,cal_acc,cal_mag";

// Gravity-vector roll candidates.
// First capture (2026-08-28, robot upright): g = (-0.35, -9.79, +0.06) -- the vertical
// axis is the IMU's y, i.e. the board is mounted rotated 90 deg (which is also why the
// device reports pitch ~= 89.6 deg while standing). So the vertical component is gy and
// the lateral axis is x or z; which of the two is lateral vs forward is what the bench
// poses decide, so BOTH are printed and neither is assumed.
// At that upright pose atan2(gx,-gy) = -2.05 deg, matching the device's own eul_roll
// (-2.06) and the "IMU reads -2 when level" note in pengu_champ_k0_105.ino.
float rollFrom(float lat, float vert) {
  return degrees(atan2(lat, -vert));
}

void printHeader() {
  DEBUG_SERIAL.println("# imu_probe v1 -- BNO055 raw logger (no motors, no wifi)");
  DEBUG_SERIAL.println("# eul_* = device Euler (x=yaw/heading, y=roll, z=pitch), deg");
  DEBUG_SERIAL.println("# g* = VECTOR_GRAVITY (m/s^2), a* = VECTOR_ACCELEROMETER (m/s^2, gravity NOT removed)");
  DEBUG_SERIAL.println("# roll_xy = atan2(gx,-gy), roll_zy = atan2(gz,-gy)  [deg; vertical axis is y on this mount]");
  DEBUG_SERIAL.println("# cal_acc must be 3 before the gravity vector is trustworthy");
  DEBUG_SERIAL.println(HDR);
}

void readSample() {
  imu::Quaternion q = bno.getQuat();
  s_q[0] = q.w(); s_q[1] = q.x(); s_q[2] = q.y(); s_q[3] = q.z();

  imu::Vector<3> e = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  s_eul[0] = e.x(); s_eul[1] = e.y(); s_eul[2] = e.z();   // same mapping the champ sketches use

  imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  s_g[0] = g.x(); s_g[1] = g.y(); s_g[2] = g.z();

  imu::Vector<3> a = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  s_a[0] = a.x(); s_a[1] = a.y(); s_a[2] = a.z();

  bno.getCalibration(&s_cal[0], &s_cal[1], &s_cal[2], &s_cal[3]);
}

void printRow() {
  DEBUG_SERIAL.print(millis());   DEBUG_SERIAL.print(',');
  DEBUG_SERIAL.print(pose_id);    DEBUG_SERIAL.print(',');
  for (int i = 0; i < 4; i++) { DEBUG_SERIAL.print(s_q[i], 5);   DEBUG_SERIAL.print(','); }
  for (int i = 0; i < 3; i++) { DEBUG_SERIAL.print(s_eul[i], 2); DEBUG_SERIAL.print(','); }
  for (int i = 0; i < 3; i++) { DEBUG_SERIAL.print(s_g[i], 4);   DEBUG_SERIAL.print(','); }
  for (int i = 0; i < 3; i++) { DEBUG_SERIAL.print(s_a[i], 4);   DEBUG_SERIAL.print(','); }
  DEBUG_SERIAL.print(rollFrom(s_g[0], s_g[1]), 2); DEBUG_SERIAL.print(',');   // atan2(gx,-gy)
  DEBUG_SERIAL.print(rollFrom(s_g[2], s_g[1]), 2); DEBUG_SERIAL.print(',');   // atan2(gz,-gy)
  for (int i = 0; i < 4; i++) {
    DEBUG_SERIAL.print(s_cal[i]);
    DEBUG_SERIAL.print(i == 3 ? '\n' : ',');
  }
}

// 2-second static average: the bench poses are held by hand, so a single sample is
// dominated by hand tremor. Prints mean and sd of each channel as a #AVG comment.
void staticAverage() {
  const int NCH = 15;                       // q4 + eul3 + g3 + a3 + roll2
  double sum[NCH], sq[NCH];
  for (int i = 0; i < NCH; i++) { sum[i] = 0.0; sq[i] = 0.0; }
  uint8_t calmin[4] = {255, 255, 255, 255};
  uint32_t t0 = millis();
  int n = 0;

  DEBUG_SERIAL.print("# AVG start pose="); DEBUG_SERIAL.print(pose_id);
  DEBUG_SERIAL.println("  hold still ...");
  while (millis() - t0 < AVG_MS) {
    readSample();
    double v[NCH] = { s_q[0], s_q[1], s_q[2], s_q[3],
                      s_eul[0], s_eul[1], s_eul[2],
                      s_g[0], s_g[1], s_g[2],
                      s_a[0], s_a[1], s_a[2],
                      rollFrom(s_g[0], s_g[1]), rollFrom(s_g[2], s_g[1]) };
    for (int i = 0; i < NCH; i++) { sum[i] += v[i]; sq[i] += v[i] * v[i]; }
    for (int i = 0; i < 4; i++) if (s_cal[i] < calmin[i]) calmin[i] = s_cal[i];
    n++;
    delay(20);                              // sample the average at 50 Hz regardless of stream rate
  }

  const char *nm[NCH] = { "qw", "qx", "qy", "qz", "eul_yaw", "eul_roll", "eul_pitch",
                          "gx", "gy", "gz", "ax", "ay", "az", "roll_xy", "roll_zy" };
  DEBUG_SERIAL.print("#AVG pose="); DEBUG_SERIAL.print(pose_id);
  DEBUG_SERIAL.print(" n="); DEBUG_SERIAL.print(n);
  for (int i = 0; i < NCH; i++) {
    double m = sum[i] / n;
    double var = sq[i] / n - m * m;
    DEBUG_SERIAL.print("  "); DEBUG_SERIAL.print(nm[i]);
    DEBUG_SERIAL.print('=');  DEBUG_SERIAL.print(m, 3);
    DEBUG_SERIAL.print("+-"); DEBUG_SERIAL.print(var > 0 ? sqrt(var) : 0.0, 3);
  }
  DEBUG_SERIAL.print("  cal_min=");
  for (int i = 0; i < 4; i++) { DEBUG_SERIAL.print(calmin[i]); DEBUG_SERIAL.print(i == 3 ? '\n' : '/'); }
}

void setup() {
  DEBUG_SERIAL.begin(115200);
  uint32_t t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 3000);

  Wire.begin();
  if (!bno.begin()) {
    DEBUG_SERIAL.println("# ERROR: BNO055 not detected (check I2C wiring / 0x28).");
    while (1);
  }
  bno.setExtCrystalUse(true);
  delay(100);
  printHeader();
  DEBUG_SERIAL.println("# stream is OFF. z=2s average (one line)  space=stream on/off  f=rate  c=cal  0-9=pose");
}

void loop() {
  if (DEBUG_SERIAL.available()) {
    char c = (char)DEBUG_SERIAL.read();
    if (c >= '0' && c <= '9') {
      pose_id = c - '0';
      DEBUG_SERIAL.print("# pose -> "); DEBUG_SERIAL.println(pose_id);
    } else if (c == ' ') {
      streaming = !streaming;
      DEBUG_SERIAL.print(streaming ? "# stream ON @ " : "# stream OFF (was ");
      DEBUG_SERIAL.print(1000 / RATES_MS[rate_idx]); DEBUG_SERIAL.println(" Hz");
    } else if (c == 'f') {
      rate_idx = (rate_idx + 1) % 4;
      DEBUG_SERIAL.print("# stream rate -> "); DEBUG_SERIAL.print(1000 / RATES_MS[rate_idx]);
      DEBUG_SERIAL.println(" Hz");
    } else if (c == 'h') {
      printHeader();
    } else if (c == 'z') {
      staticAverage();
    } else if (c == 'c') {
      uint8_t cs, cg, ca, cm;
      bno.getCalibration(&cs, &cg, &ca, &cm);
      DEBUG_SERIAL.print("# cal sys/gyro/acc/mag = ");
      DEBUG_SERIAL.print(cs); DEBUG_SERIAL.print('/'); DEBUG_SERIAL.print(cg); DEBUG_SERIAL.print('/');
      DEBUG_SERIAL.print(ca); DEBUG_SERIAL.print('/'); DEBUG_SERIAL.println(cm);
    }
  }

  if (streaming && millis() - t_last >= RATES_MS[rate_idx]) {
    t_last = millis();
    readSample();
    printRow();
  }
}
