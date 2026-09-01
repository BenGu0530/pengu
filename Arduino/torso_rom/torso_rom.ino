// torso_rom.ino — how far can the torso actually roll before it hits the legs?
//
// Why this matters: the sim's kappa PID commands a torso JOINT swing of +-22 deg
// (kappa=2, c6) to +-27 deg (kappa=0, c3), and the champ sketches clamp at +-45 deg.
// Ben's note on pengu_champ_k0_105 says the torso collides with the legs beyond ~+-10.
// If the mechanical range really is +-10, then no controller of any kind can produce the
// 21 deg of torso world roll the sim's kappa=2 relies on, and that is a geometry result,
// not a motor or a tuning result. So: measure the range before running the walk.
//
// SAFE BY CONSTRUCTION: the torso runs in CURRENT-BASED POSITION control with a gentle
// current cap, so when it reaches the stop the motor simply cannot push harder -- it
// stalls against the limit at a torque you chose, instead of driving into it. The ramp
// is slow (RAMP_DPS deg/s) and aborts the moment the joint stops following its goal.
//
// The legs' position changes the clearance, so the routine measures the range at BOTH
// hip offsets that matter: hips at 0 (the design neutral, where the COM ratio and the
// PID are defined) and hips at the walking lean.
//
// Serial 115200. Commands:
//   b   A-reconstruction check at BOTH hip offsets (the main event)
//   k   A-reconstruction check at the current hip offset only
//   r   ROM routine: for hip_off in {0, WALK_HIP_OFF}, ramp + then -, report 4 limits
//   +   single slow ramp in the + direction at the current hip offset
//   -   single slow ramp in the - direction
//   0   put hips at 0 and hold          2   put hips at WALK_HIP_OFF and hold
//   c   return torso to zero and hold
//   q   abort: stop the ramp, hold where it is
//   p   health of all five motors
//
// Output: CSV rows (label,hip_off,dir,t,goal,pos,err,mA) plus a #LIMIT line per ramp.

#include <DynamixelShield.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

DynamixelShield dxl;
const float DXL_PROTOCOL_VERSION = 2.0;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

const uint8_t XM_TORSO_ROLL  = 0;
const uint8_t XM_RIGHT_HIP   = 1;
const uint8_t XM_LEFT_HIP    = 2;
const uint8_t XM_RIGHT_SLIDE = 3;
const uint8_t XM_LEFT_SLIDE  = 4;
const uint8_t LOCK_IDS[4] = {XM_RIGHT_HIP, XM_LEFT_HIP, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};

const float CURRENT_UNIT_MA = 2.69f;      // XM430 PRESENT_CURRENT / GOAL_CURRENT unit
const float NM_PER_AMP      = 1.78f;      // nominal: 4.1 N.m at 2.3 A

// ---- test parameters ----------------------------------------------------------------
const float CUR_CAP_MA   = 800.0f;   // torso push limit (~1.4 N.m). Raise if it will not
                                     // even lift its own weight; lower if you want gentler.
const float RAMP_DPS     = 3.0f;     // slow: 3 deg/s
const float MAX_TEST_DEG = 40.0f;    // never command beyond this
const float STALL_ERR    = 4.0f;     // goal-present error that counts as "stopped"
const float STALL_HOLD_S = 0.6f;     // ... sustained this long
const float WALK_HIP_OFF = 20.0f;    // c6 champion hip_off
const float BACKOFF_DEG  = 3.0f;     // ease off this much after finding a limit

float torso_home = 0, lock_home[4] = {0, 0, 0, 0};
float hip_off_now = 0;

float zeroExtended(float cur) {
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  return cur - ((phys > 180.0f) ? phys - 360.0f : phys);
}

float readCurrent_mA(uint8_t id) {
  return (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, id) * CURRENT_UNIT_MA;
}

float torsoPos() { return dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - torso_home; }

// hips: R gets +off, L gets -off (champ sketch convention); cranks stay at zero
void setLocks(float off) {
  dxl.setGoalPosition(XM_RIGHT_HIP,   lock_home[0] + off, UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_HIP,    lock_home[1] - off, UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_SLIDE, lock_home[2], UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_SLIDE,  lock_home[3], UNIT_DEGREE);
}

void gotoHipOff(float off) {
  float from = hip_off_now;
  int steps = 100;                                   // 2 s ramp
  for (int k = 1; k <= steps; k++) {
    setLocks(from + (off - from) * k / (float)steps);
    delay(20);
  }
  hip_off_now = off;
  DEBUG_SERIAL.print("# hips at "); DEBUG_SERIAL.print(off, 1); DEBUG_SERIAL.println(" deg");
}

void torsoToZero() {
  float from = torsoPos();
  int steps = 100;
  for (int k = 1; k <= steps; k++) {
    dxl.setGoalPosition(XM_TORSO_ROLL, torso_home + from * (1.0f - k / (float)steps), UNIT_DEGREE);
    delay(20);
  }
  DEBUG_SERIAL.print("# torso back to zero (pos="); DEBUG_SERIAL.print(torsoPos(), 2);
  DEBUG_SERIAL.println(")");
}

// Slow ramp in one direction until the joint stops following. Returns the limit angle.
float rampToLimit(int dir) {
  DEBUG_SERIAL.print("# ramp dir="); DEBUG_SERIAL.print(dir > 0 ? "+" : "-");
  DEBUG_SERIAL.print(" at hip_off="); DEBUG_SERIAL.print(hip_off_now, 1);
  DEBUG_SERIAL.print("  cap="); DEBUG_SERIAL.print(CUR_CAP_MA, 0); DEBUG_SERIAL.println(" mA");

  torsoToZero();
  float goal = 0, stallT = 0, limit = 0;
  bool tripped = false;
  uint32_t prev = millis();
  while (fabsf(goal) < MAX_TEST_DEG) {
    uint32_t now = millis();
    float dt = (now - prev) / 1000.0f;
    prev = now;
    goal += dir * RAMP_DPS * dt;
    dxl.setGoalPosition(XM_TORSO_ROLL, torso_home + goal, UNIT_DEGREE);

    float pos = torsoPos();
    float err = goal - pos;
    float mA  = readCurrent_mA(XM_TORSO_ROLL);

    DEBUG_SERIAL.print("d,"); DEBUG_SERIAL.print(hip_off_now, 0); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(dir);  DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print((now % 1000000) / 1000.0f, 3); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(goal, 2); DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(pos, 2);
    DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(err, 2); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.println(mA, 0);

    if (fabsf(err) > STALL_ERR) {
      stallT += dt;
      if (stallT >= STALL_HOLD_S) { limit = pos; tripped = true; break; }
    } else {
      stallT = 0;
    }
    if (DEBUG_SERIAL.available() && DEBUG_SERIAL.read() == 'q') {
      DEBUG_SERIAL.println("# aborted by user");
      torsoToZero();
      return NAN;
    }
    delay(20);
  }
  if (!tripped) {
    limit = torsoPos();
    DEBUG_SERIAL.print("#LIMIT hip_off="); DEBUG_SERIAL.print(hip_off_now, 0);
    DEBUG_SERIAL.print(" dir="); DEBUG_SERIAL.print(dir > 0 ? "+" : "-");
    DEBUG_SERIAL.print(" NO STOP FOUND within "); DEBUG_SERIAL.print(MAX_TEST_DEG, 0);
    DEBUG_SERIAL.print(" deg (reached "); DEBUG_SERIAL.print(limit, 1); DEBUG_SERIAL.println(")");
  } else {
    // ease off the stop before reporting, so it does not sit pressed against the legs
    dxl.setGoalPosition(XM_TORSO_ROLL, torso_home + limit - dir * BACKOFF_DEG, UNIT_DEGREE);
    delay(300);
    DEBUG_SERIAL.print("#LIMIT hip_off="); DEBUG_SERIAL.print(hip_off_now, 0);
    DEBUG_SERIAL.print(" dir="); DEBUG_SERIAL.print(dir > 0 ? "+" : "-");
    DEBUG_SERIAL.print(" stop_at="); DEBUG_SERIAL.print(limit, 1);
    DEBUG_SERIAL.print(" deg  mA="); DEBUG_SERIAL.print(readCurrent_mA(XM_TORSO_ROLL), 0);
    DEBUG_SERIAL.print("  (cap "); DEBUG_SERIAL.print(CUR_CAP_MA, 0);
    DEBUG_SERIAL.print(" mA ~ "); DEBUG_SERIAL.print(CUR_CAP_MA * 0.001f * NM_PER_AMP, 2);
    DEBUG_SERIAL.println(" N.m)");
  }
  torsoToZero();
  return limit;
}

// ---- A-reconstruction check ---------------------------------------------------------
// Hanging with hips and cranks locked, the lower body cannot move, so the hip-axis roll A
// is CONSTANT. The firmware reconstructs it as A_est = imu_roll - S_TILT * J. If that
// reconstruction is sound, sweeping the torso across its range must leave A_est flat: a
// horizontal line. Whatever it does instead IS the reconstruction error, measured rather
// than argued -- and the fitted slope d(imu_roll)/dJ is the value S_TILT should have
// (expected +-1; the sim measures its own equivalent at startup instead of hardcoding it).
//
// The kappa=2 control law amplifies an error in A by d(cmd)/dA = (k-1) + kp*k = 5, so read
// the residual below multiplied by 5 to get the torso command error it would cause.
void sweepA(float span) {
  DEBUG_SERIAL.print("# A-check sweep +-"); DEBUG_SERIAL.print(span, 0);
  DEBUG_SERIAL.print(" deg at hip_off="); DEBUG_SERIAL.println(hip_off_now, 1);
  torsoToZero();

  // least squares of roll on J, plus min/max of A_est = roll - slope_sign*J
  double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  float legJ[400], legR[400]; int nlog = 0;

  float goal = 0;
  int dir = -1;                      // go negative first, then all the way positive
  uint32_t prev = millis();
  int leg = 0;
  while (leg < 2) {
    uint32_t now = millis();
    float dt = (now - prev) / 1000.0f; prev = now;
    goal += dir * RAMP_DPS * dt;
    if (dir < 0 && goal <= -span) { dir = +1; leg = 1; }
    else if (dir > 0 && goal >= span) { leg = 2; }
    dxl.setGoalPosition(XM_TORSO_ROLL, torso_home + goal, UNIT_DEGREE);

    imu::Vector<3> e = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
    imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
    float roll = e.y(), pitch = e.z();
    float J = torsoPos();
    float mA = readCurrent_mA(XM_TORSO_ROLL);

    n += 1; sx += J; sy += roll; sxx += (double)J * J; sxy += (double)J * roll; syy += (double)roll * roll;
    if (nlog < 400) { legJ[nlog] = J; legR[nlog] = roll; nlog++; }

    DEBUG_SERIAL.print("k,"); DEBUG_SERIAL.print(hip_off_now, 0); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(goal, 2); DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(J, 2);
    DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(roll, 2);
    DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(pitch, 2);
    DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(g.x(), 3);
    DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(g.y(), 3);
    DEBUG_SERIAL.print(','); DEBUG_SERIAL.print(g.z(), 3);
    DEBUG_SERIAL.print(','); DEBUG_SERIAL.println(mA, 0);

    if (DEBUG_SERIAL.available() && DEBUG_SERIAL.read() == 'q') {
      DEBUG_SERIAL.println("# aborted"); torsoToZero(); return;
    }
    delay(50);
  }
  torsoToZero();

  double den = n * sxx - sx * sx;
  if (den < 1e-9 || n < 10) { DEBUG_SERIAL.println("#ACHK not enough data"); return; }
  double slope = (n * sxy - sx * sy) / den;
  double icept = (sy - slope * sx) / n;
  double sst = syy - sy * sy / n;
  double sse = syy - icept * sy - slope * sxy;
  double r2 = (sst > 1e-9) ? 1.0 - sse / sst : 0.0;

  // A_est with S_TILT = sign(slope): should be constant if the reconstruction is sound
  float S = (slope >= 0) ? 1.0f : -1.0f;
  double amin = 1e9, amax = -1e9, asum = 0, asq = 0;
  for (int i = 0; i < nlog; i++) {
    double Ae = legR[i] - S * legJ[i];
    if (Ae < amin) amin = Ae; if (Ae > amax) amax = Ae;
    asum += Ae; asq += Ae * Ae;
  }
  double amean = asum / nlog;
  double asd = sqrt(fmax(0.0, asq / nlog - amean * amean));

  DEBUG_SERIAL.print("#ACHK hip_off="); DEBUG_SERIAL.print(hip_off_now, 0);
  DEBUG_SERIAL.print(" n="); DEBUG_SERIAL.print((int)n);
  DEBUG_SERIAL.print(" | d(roll)/dJ="); DEBUG_SERIAL.print(slope, 3);
  DEBUG_SERIAL.print(" intercept="); DEBUG_SERIAL.print(icept, 2);
  DEBUG_SERIAL.print(" R2="); DEBUG_SERIAL.print(r2, 4);
  DEBUG_SERIAL.print(" -> S_TILT should be "); DEBUG_SERIAL.print(S, 0);
  DEBUG_SERIAL.print(" | A_est mean="); DEBUG_SERIAL.print(amean, 2);
  DEBUG_SERIAL.print(" sd="); DEBUG_SERIAL.print(asd, 2);
  DEBUG_SERIAL.print(" range="); DEBUG_SERIAL.print(amax - amin, 2);
  DEBUG_SERIAL.print(" deg  => kappa=2 torso command error up to ");
  DEBUG_SERIAL.print(5.0 * (amax - amin), 1); DEBUG_SERIAL.println(" deg");
}

void setup() {
  DEBUG_SERIAL.begin(115200);
  uint32_t t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 3000);

  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  // locks: plain extended-position, full torque -- they must not move
  for (int i = 0; i < 4; i++) {
    uint8_t id = LOCK_IDS[i];
    if (!dxl.ping(id)) { DEBUG_SERIAL.print("# No response ID "); DEBUG_SERIAL.println(id); }
    float boot = dxl.getPresentPosition(id, UNIT_DEGREE);
    lock_home[i] = zeroExtended(boot);
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.torqueOn(id);
    dxl.setGoalPosition(id, boot, UNIT_DEGREE);
    DEBUG_SERIAL.print("# lock ID "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print(" boot="); DEBUG_SERIAL.print(boot, 2);
    DEBUG_SERIAL.print(" home="); DEBUG_SERIAL.println(lock_home[i], 2);
  }

  // torso: CURRENT-BASED position control, so it cannot push past CUR_CAP_MA
  if (!dxl.ping(XM_TORSO_ROLL)) DEBUG_SERIAL.println("# No response ID 0 (torso)");
  float boot = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE);
  torso_home = zeroExtended(boot);
  dxl.torqueOff(XM_TORSO_ROLL);
  dxl.setOperatingMode(XM_TORSO_ROLL, OP_CURRENT_BASED_POSITION);
  dxl.writeControlTableItem(PROFILE_VELOCITY, XM_TORSO_ROLL, 0);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, XM_TORSO_ROLL, 0);
  dxl.writeControlTableItem(GOAL_CURRENT, XM_TORSO_ROLL, (int)(CUR_CAP_MA / CURRENT_UNIT_MA));
  dxl.torqueOn(XM_TORSO_ROLL);
  dxl.setGoalPosition(XM_TORSO_ROLL, boot, UNIT_DEGREE);
  DEBUG_SERIAL.print("# torso ID 0 boot="); DEBUG_SERIAL.print(boot, 2);
  DEBUG_SERIAL.print(" home="); DEBUG_SERIAL.print(torso_home, 2);
  DEBUG_SERIAL.print("  current-based position, cap "); DEBUG_SERIAL.print(CUR_CAP_MA, 0);
  DEBUG_SERIAL.println(" mA");

  Wire.begin();
  if (!bno.begin()) DEBUG_SERIAL.println("# WARNING: BNO055 not detected -- k sweep unusable");
  else { bno.setExtCrystalUse(true); DEBUG_SERIAL.println("# BNO055 ok"); }

  DEBUG_SERIAL.println("label,hip_off,dir,t,goal,pos,err,mA");
  DEBUG_SERIAL.println("k,hip_off,goal,J,imu_roll,imu_pitch,gx,gy,gz,mA   <- A-check rows");
  DEBUG_SERIAL.println("# HANG THE ROBOT. b=A-check both offsets  k=A-check here  r=ROM routine  0/2=hips  c=center  q=abort");
}

void loop() {
  if (!DEBUG_SERIAL.available()) { delay(5); return; }
  char c = (char)DEBUG_SERIAL.read();
  if (c == '\n' || c == '\r') return;
  // echo every keypress: a mistyped key should say so, not sit there silently
  DEBUG_SERIAL.print("# key '"); DEBUG_SERIAL.print(c); DEBUG_SERIAL.println("'");

  if (c == 'r') {
    float offs[2] = {0.0f, WALK_HIP_OFF};
    for (int i = 0; i < 2; i++) {
      gotoHipOff(offs[i]);
      delay(500);
      if (isnan(rampToLimit(+1))) return;
      delay(500);
      if (isnan(rampToLimit(-1))) return;
    }
    gotoHipOff(0.0f);
    DEBUG_SERIAL.println("# routine done");
  } else if (c == 'k') {
    sweepA(20.0f);
  } else if (c == 'b') {
    for (int i = 0; i < 2; i++) { gotoHipOff(i == 0 ? 0.0f : WALK_HIP_OFF); delay(400); sweepA(20.0f); }
    gotoHipOff(0.0f);
    DEBUG_SERIAL.println("# A-check done at both hip offsets");
  } else if (c == '+') {
    rampToLimit(+1);
  } else if (c == '-') {
    rampToLimit(-1);
  } else if (c == '0') {
    gotoHipOff(0.0f);
  } else if (c == '2') {
    gotoHipOff(WALK_HIP_OFF);
  } else if (c == 'c') {
    torsoToZero();
  } else if (c == 'q') {
    DEBUG_SERIAL.println("# idle");
  } else if (c != 'p') {
    DEBUG_SERIAL.println("#   unknown -- b=A-check both offsets  k=A-check here  r=ROM routine"
                         "  +/-=single ramp  0/2=hips  c=center  q=abort  p=health");
  }
  if (c == 'p') {
    const uint8_t ALL[5] = {XM_TORSO_ROLL, XM_RIGHT_HIP, XM_LEFT_HIP, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};
    for (int i = 0; i < 5; i++) {
      uint8_t id = ALL[i];
      bool ok = dxl.ping(id);
      DEBUG_SERIAL.print("# ID "); DEBUG_SERIAL.print(id);
      DEBUG_SERIAL.print(ok ? "  ping OK" : "  PING FAIL");
      DEBUG_SERIAL.print("  pos=");    DEBUG_SERIAL.print(dxl.getPresentPosition(id, UNIT_DEGREE), 1);
      DEBUG_SERIAL.print("  mA=");     DEBUG_SERIAL.print(readCurrent_mA(id), 0);
      DEBUG_SERIAL.print("  torque="); DEBUG_SERIAL.print(dxl.readControlTableItem(TORQUE_ENABLE, id));
      DEBUG_SERIAL.print("  hwErr=");  DEBUG_SERIAL.println(dxl.readControlTableItem(HARDWARE_ERROR_STATUS, id));
    }
  }
}
