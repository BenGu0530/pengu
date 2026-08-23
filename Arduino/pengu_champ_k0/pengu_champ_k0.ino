// pengu_champ_k0.ino — GRID-4 c3 champion gait (COM-1.31 body, Gait 1: kappa=0,
// torso held WORLD-UPRIGHT by feedback), K=5-verified in sim at mu=0.1.
// A/B counterpart to pengu_champ (kappa=2) on the same slippery surface.
//
// Self-contained: no WiFi tabs needed. Serial commands:
//   r = READY (slides -> hips -> torso to home)     w = WALK     q = IDLE (hold)
//   s = print IMU roll + torso joint angle (for the S_TILT sign check)
// Optional: set AUTO_WALK true to run READY -> wait -> WALK on power-up (untethered).
//
// ---------------- BEFORE FIRST UNTETHERED RUN (hang the robot!) ----------------
// 1. S_TILT sign: send 's', rotate torso motor + a few deg by hand; if imu_roll moves
//    the SAME way keep S_TILT=+1, opposite -> -1.  WRONG SIGN = POSITIVE FEEDBACK,
//    the torso will slam itself over. Verify while hanging.
// 2. Hip direction: in WALK the hips must swing FORWARD from the leaned-forward
//    ready pose. If they swing backward, flip the two hip signs in the setGoal block.
// 3. Legs should alternate extend/retract (same convention as the old pengu.ino).
// XM430 note: Profile Velocity/Acceleration = 0 (unlimited), else 1.67 Hz targets
// get slew-limited into mush.

#include <DynamixelShield.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <WiFiNINA.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

// ===================== Champion gait (sim units: deg, Hz) =====================
// Ice champion (mu=0.1): freq 1.61 phi 330 leg 115 hip 28 off 10 (K5 net 0.164, head 0.99)
// Grippy-floor tier (mu=0.3, its home turf; WARNING freq~2Hz at the sim2real edge):
//   1.97 / 20 / 125 / 28 / 10  (K5 net 0.490)
// old names/units kept so webpage.ino + wireless.ino tabs work unchanged (rad where noted)
float p_legFreq  = 1.61f;                     // [Hz]
float p_legAmp   = 115.0f * PI / 180.0f;      // [rad] crank amplitude (unipolar)
float p_hipAmp   = 28.0f  * PI / 180.0f;      // [rad] hip half-rectified swing
float p_hipPhiD  = 330.0f;                    // [deg] hip-vs-leg phase offset
float p_hipOffD  = 10.0f;                     // [deg] symmetric forward-pitch offset

// torso kappa-PID (sim TorsoKappaPID values)
float p_kappa = 0.0f, p_kp = 2.0f, p_ki = 0.1f;   // kappa=0: torso counter-rotates to stay world-upright
const float TORSO_CLAMP_DEG = 45.0f;
float S_TILT = +1.0f;                  // <-- VERIFY (check #1 above)

// staged start (sim-validated: no hip_off step-shove)
const float T_RAMP = 4.0f, T_SETTLE = 6.0f, T_BLEND = 4.0f;

// autostart for untethered runs: power on -> READY -> AUTO_WAIT s -> WALK
const bool  AUTO_WALK = false;
const float AUTO_WAIT = 5.0f;          // [s] time to place the robot after READY

// ===================== Motor IDs (same rig as pengu.ino) =====================
const uint8_t XM_LEFT_SLIDE  = 4;
const uint8_t XM_RIGHT_SLIDE = 3;
const uint8_t XM_LEFT_HIP    = 2;
const uint8_t XM_RIGHT_HIP   = 1;
const uint8_t XM_TORSO_ROLL  = 0;
const uint8_t MOTOR_IDS[] = {XM_LEFT_SLIDE, XM_RIGHT_SLIDE, XM_LEFT_HIP, XM_RIGHT_HIP, XM_TORSO_ROLL};
const int MOTOR_COUNT = 5;

DynamixelShield dxl;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

enum RobotState { STATE_IDLE, STATE_READY_SLIDE, STATE_READY_HIP, STATE_READY_TORSO,
                  STATE_AUTO_WAIT, STATE_WALK };
RobotState robot_state = STATE_IDLE;

float   imu_roll = 0, imu_pitch = 0, imu_yaw = 0;
float   imu_ax = 0, imu_ay = 0, imu_az = 0;
uint8_t cal_sys = 0, cal_gyro = 0, cal_accel = 0, cal_mag = 0;
bool    wifi_active = false;
float   home_deg[MOTOR_COUNT];
unsigned long walk_start_ms = 0, autowait_ms = 0;
float torso_iErr = 0.0f;
bool  stream_dbg = false;          // 't' toggles a 5 Hz roll/joint/axis stream (hand-rock test)

const float READY_STEP = 1.0f, READY_STEP_TORSO = 1.5f, ARRIVE_THRESH = 1.0f;

int idxOf(uint8_t id) {
  for (int i = 0; i < MOTOR_COUNT; i++) if (MOTOR_IDS[i] == id) return i;
  return -1;
}
float shortestDelta(float cur, float tgt) {
  float d = tgt - cur;
  while (d >  180.0f) d -= 360.0f;
  while (d < -180.0f) d += 360.0f;
  return d;
}
void stepMotorToward(uint8_t id, float target, float step) {
  float cur = dxl.getPresentPosition(id, UNIT_DEGREE);
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  float d = shortestDelta(phys, target);
  dxl.setGoalPosition(id, cur + ((fabsf(d) <= step) ? d : (d > 0 ? step : -step)), UNIT_DEGREE);
}
bool arrivedAt(uint8_t id, float target) {
  float phys = fmod(dxl.getPresentPosition(id, UNIT_DEGREE), 360.0f);
  if (phys < 0) phys += 360.0f;
  return fabsf(shortestDelta(phys, target)) < ARRIVE_THRESH;
}

void setup() {
  DEBUG_SERIAL.begin(115200);
  unsigned long t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 3000);   // don't block untethered boots

  dxl.begin(1000000);
  dxl.setPortProtocolVersion(2.0);
  for (int i = 0; i < MOTOR_COUNT; i++) {
    uint8_t id = MOTOR_IDS[i];
    if (!dxl.ping(id)) { DEBUG_SERIAL.print("No response ID "); DEBUG_SERIAL.println(id); }
    home_deg[i] = dxl.getPresentPosition(id, UNIT_DEGREE);
    dxl.torqueOff(id);
    // EXTENDED position: accepts negative / >360 deg goals. In plain OP_POSITION a goal
    // like home-20 with home near 0 deg is silently REJECTED (suspected hip/torso no-move).
    dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
    // unlimited profile so 1.67 Hz targets are tracked, not slew-limited
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.torqueOn(id);
    dxl.setGoalPosition(id, home_deg[i], UNIT_DEGREE);
    DEBUG_SERIAL.print("ID "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print(" home = "); DEBUG_SERIAL.println(home_deg[i], 2);
  }

  if (!bno.begin()) { DEBUG_SERIAL.println("ERROR: BNO055 not detected."); while (1); }
  bno.setExtCrystalUse(true);

  begin_wifi();                                  // webpage + /cmd + /data (wireless.ino tab)

  DEBUG_SERIAL.println("pengu_champ_k0 (Gait 1) ready. r=Ready w=Walk q=Idle s=SignCheck (serial or wifi)");
  if (AUTO_WALK) { robot_state = STATE_READY_SLIDE; DEBUG_SERIAL.println("AUTO: -> READY"); }
}

void update_imu() {
  imu::Vector<3> e = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  imu::Vector<3> a = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  bno.getCalibration(&cal_sys, &cal_gyro, &cal_accel, &cal_mag);
  imu_yaw = e.x(); imu_roll = e.y(); imu_pitch = e.z();   // same mapping as pengu.ino
  imu_ax = a.x(); imu_ay = a.y(); imu_az = a.z();
}

// ===================== WALK: c6 champion =====================
void run_walk() {
  float t = (millis() - walk_start_ms) / 1000.0f;

  // staged start: ramp hip_off -> settle -> blend the oscillation in
  float off_deg = p_hipOffD * constrain(t / T_RAMP, 0.0f, 1.0f);
  float alpha   = constrain((t - T_RAMP - T_SETTLE) / T_BLEND, 0.0f, 1.0f);
  float phase   = (t > T_RAMP + T_SETTLE) ? 2.0f * PI * p_legFreq * (t - T_RAMP - T_SETTLE) : 0.0f;

  // legs: unipolar antiphase  (sim: crank = 0.5*amp*(1+sin))
  float A_leg = p_legAmp * 180.0f / PI;
  float magL = alpha * 0.5f * A_leg * (1.0f + sinf(phase));
  float magR = alpha * 0.5f * A_leg * (1.0f + sinf(phase + PI));

  // hips: half-rectified antiphase + phase offset + symmetric forward offset
  float A_hip = p_hipAmp * 180.0f / PI;
  float phi = p_hipPhiD * PI / 180.0f;
  float hipL_deg = off_deg + alpha * A_hip * max(0.0f, sinf(phase + PI + phi));
  float hipR_deg = off_deg + alpha * A_hip * max(0.0f, sinf(phase + phi));

  // torso: kappa=2 feedback — target torso WORLD roll = kappa * hip-axis roll
  float J_deg  = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - home_deg[idxOf(XM_TORSO_ROLL)];
  float axis   = imu_roll - S_TILT * J_deg;
  float err    = p_kappa * axis - imu_roll;
  torso_iErr   = constrain(torso_iErr + err * 0.02f, -20.0f, 20.0f);
  float torso_deg = alpha * constrain(
      (p_kappa - 1.0f) * axis / S_TILT + p_kp * err + p_ki * torso_iErr,
      -TORSO_CLAMP_DEG, TORSO_CLAMP_DEG);

  // sign conventions inherited from pengu.ino (L negative / R positive); see check #2
  dxl.setGoalPosition(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)]  - magL,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)] + magR,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_HIP,    home_deg[idxOf(XM_LEFT_HIP)]    - hipL_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_HIP,   home_deg[idxOf(XM_RIGHT_HIP)]   + hipR_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_TORSO_ROLL,  home_deg[idxOf(XM_TORSO_ROLL)]  + torso_deg, UNIT_DEGREE);

  // 1 Hz debug: what is actually being commanded (catch out-of-range / alpha timing)
  static unsigned long dbg = 0;
  if (millis() - dbg > 1000) { dbg = millis();
    DEBUG_SERIAL.print("t=");      DEBUG_SERIAL.print(t, 1);
    DEBUG_SERIAL.print(" a=");     DEBUG_SERIAL.print(alpha, 2);
    DEBUG_SERIAL.print(" hipL->"); DEBUG_SERIAL.print(home_deg[idxOf(XM_LEFT_HIP)] - hipL_deg, 1);
    DEBUG_SERIAL.print(" hipR->"); DEBUG_SERIAL.print(home_deg[idxOf(XM_RIGHT_HIP)] + hipR_deg, 1);
    DEBUG_SERIAL.print(" torso->");DEBUG_SERIAL.print(home_deg[idxOf(XM_TORSO_ROLL)] + torso_deg, 1);
    DEBUG_SERIAL.print(" roll=");  DEBUG_SERIAL.print(imu_roll, 1);
    DEBUG_SERIAL.print(" axis=");  DEBUG_SERIAL.println(axis, 1);
  }
}

void loop() {
  update_imu();

  char cmd = 0;
  if (wifi_active) cmd = update_wifi();          // /cmd?key=X from the webpage
  if (DEBUG_SERIAL.available()) cmd = (char)DEBUG_SERIAL.read();
  switch (cmd) {
    case 'r': robot_state = STATE_READY_SLIDE; DEBUG_SERIAL.println("-> READY"); break;
    case 'w':
      if (robot_state == STATE_IDLE) {
        walk_start_ms = millis(); torso_iErr = 0;
        robot_state = STATE_WALK; DEBUG_SERIAL.println("-> WALK (staged start)");
      } else DEBUG_SERIAL.println("Go READY first.");
      break;
    case 'q': robot_state = STATE_IDLE; DEBUG_SERIAL.println("-> IDLE"); break;
    case 's': {
      float J = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - home_deg[idxOf(XM_TORSO_ROLL)];
      DEBUG_SERIAL.print("imu_roll="); DEBUG_SERIAL.print(imu_roll, 2);
      DEBUG_SERIAL.print("  torso_joint="); DEBUG_SERIAL.print(J, 2);
      DEBUG_SERIAL.println("  (rotate torso by hand: same direction -> S_TILT=+1, opposite -> -1)");
      break;
    }
    case 't': stream_dbg = !stream_dbg;
      DEBUG_SERIAL.println(stream_dbg ? "stream ON (rock the body by hand)" : "stream OFF");
      break;
  }

  static unsigned long strm = 0;
  if (stream_dbg && millis() - strm > 200) { strm = millis();
    float J = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - home_deg[idxOf(XM_TORSO_ROLL)];
    DEBUG_SERIAL.print("roll="); DEBUG_SERIAL.print(imu_roll, 1);
    DEBUG_SERIAL.print(" joint="); DEBUG_SERIAL.print(J, 1);
    DEBUG_SERIAL.print(" axis="); DEBUG_SERIAL.println(imu_roll - S_TILT * J, 1);
  }

  switch (robot_state) {
    case STATE_IDLE: break;
    case STATE_READY_SLIDE: {
      stepMotorToward(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)],  READY_STEP);
      stepMotorToward(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)], READY_STEP);
      if (arrivedAt(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)]) &&
          arrivedAt(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)])) {
        robot_state = STATE_READY_HIP; DEBUG_SERIAL.println("Slides done -> hips");
      }
      break;
    }
    case STATE_READY_HIP: {
      stepMotorToward(XM_LEFT_HIP,  home_deg[idxOf(XM_LEFT_HIP)],  READY_STEP);
      stepMotorToward(XM_RIGHT_HIP, home_deg[idxOf(XM_RIGHT_HIP)], READY_STEP);
      if (arrivedAt(XM_LEFT_HIP,  home_deg[idxOf(XM_LEFT_HIP)]) &&
          arrivedAt(XM_RIGHT_HIP, home_deg[idxOf(XM_RIGHT_HIP)])) {
        robot_state = STATE_READY_TORSO; DEBUG_SERIAL.println("Hips done -> torso");
      }
      break;
    }
    case STATE_READY_TORSO:
      stepMotorToward(XM_TORSO_ROLL, home_deg[idxOf(XM_TORSO_ROLL)], READY_STEP_TORSO);
      if (arrivedAt(XM_TORSO_ROLL, home_deg[idxOf(XM_TORSO_ROLL)])) {
        if (AUTO_WALK) { autowait_ms = millis(); robot_state = STATE_AUTO_WAIT;
                         DEBUG_SERIAL.println("READY done -> auto-wait"); }
        else           { robot_state = STATE_IDLE; DEBUG_SERIAL.println("READY done -> IDLE"); }
      }
      break;
    case STATE_AUTO_WAIT:
      if (millis() - autowait_ms > (unsigned long)(AUTO_WAIT * 1000)) {
        walk_start_ms = millis(); torso_iErr = 0;
        robot_state = STATE_WALK; DEBUG_SERIAL.println("AUTO -> WALK");
      }
      break;
    case STATE_WALK: run_walk(); break;
  }

  delay(20);   // ~50 Hz
}
