// pengu_k2_com105_mu05_wifi.ino — WiFi build; every command has a button at 192.168.4.1.
//
// ---------------------------------------------------------------------------------------
// pengu_k2_com105_mu05.ino — kappa=2 on the COM 1.05 body, gaits for mu = 0.5.
// GRID-4 config c4.
//
// >>> HARDWARE CHANGE REQUIRED <<<  COM 1.05 = counterweight slid DOWN ~86 mm along the
// torso mast from the 1.31 build (sim slide -86.05 mm, mass held at 2.2724 kg).
//
// Written after the mu=0.5 kappa=0 set failed on hardware: Ben reported that hip_off 10
// leaves the robot sitting BACKWARDS, which matches the firmware's own note that
// HIP_REST_DEG is 10 "or the robot tips backward" -- at hip_off 10 the walk adds no
// forward lean at all over the rest pose. Every gait here therefore leans 30 deg or more,
// and every one has a crank demand inside the ~430 deg/s ceiling measured on this robot.
//
//   'n'  1.56 / 270 / 85 / 28 / 30   (boot default)  crank 417, torso cmd +-14.8 deg
//   'm'  1.42 / 330 / 85 / 28 / 40                   crank 379, torso cmd +-33.5 deg
//   'j'  1.36 / 260 / 95 / 28 / 50                   crank 406, torso cmd +-35.3 deg
//
// 'n' is the one to start with: it is the only one whose torso command also fits inside
// the mechanical travel. The stop measured on this robot is +30.9 deg at hip_off 0 and
// +27.8 at hip_off 20, about -0.155 deg per degree of lean, so roughly +26 at lean 30,
// +25 at 40 and +23 at 50. 'n' asks for +-14.8 (11 deg of margin); 'm' and 'j' ask for
// +-33 to +-35 and will be clipped by TORSO_CLAMP_DEG = 25. They are here in case 30 deg
// of lean still is not enough to keep the robot off its heels, not because they are
// expected to reproduce the simulation.
//
// Why kappa=2 rather than the kappa=0 set that failed: on this body and this surface the
// reachable kappa=2 gaits are 2.2-2.7x faster than the reachable kappa=0 ones at every
// lean (0.246 vs 0.111 at lean 30, 0.240 vs 0.094 at 40, 0.226 vs 0.085 at 50, sweep
// net_fwd). kappa=0 only competes here if the crank envelope is ignored.
//
// Sim reference, mu=0.5, kappa=2, COM 1.05 (grid5 staged protocol, 10 s window):
//
//                        n (lean 30)  m (lean 40)  j (lean 50)
//   speed                0.141        0.141        0.227  m/s
//   torso world roll     14.5         28.2         34.2   deg rms
//   hip-axis roll A      6.8          15.1         18.9   deg rms
//   torso command amp    29.5         66.9         70.5   deg
//   torso torque p95     1.05         2.77         2.04   N.m   (stall 4.1)
//   torso >=90% stall    0.0%         1.6%         0.0%
//
// The sweep (GRID-4 protocol) rates these at 0.246 / 0.240 / 0.226; the numbers above come
// from the grid5 staged start, which is what the firmware actually does.
//
// Everything else inherited: kp = 0.5, no pacing delay (57-80 Hz), integral on the
// measured period, TORSO_CLAMP_DEG = 25, leg POSITION_P_GAIN booting at 1600.
//
// Commands:  r = READY   w = WALK   q = IDLE   0 / 2 = kappa   n / m / j = gait
//            t = telemetry   4 / 5 / 6 / 7 = kp   g = leg P gain   p = health.
//            Over WiFi every one of these has a button on the page.

#include <DynamixelShield.h>
#include <WiFiNINA.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

// ===================== Champion gait (sim units: deg, Hz) =====================
// Hardware-safe tier: freq 1.67 phi 340 leg 95 hip 24 off 20  (K5 net 0.376, straight)
// Faster tier (swap in after it walks): 1.85 / 280 / 95 / 28 / 10 (K5 net 0.444)
// old names/units kept so webpage.ino + wireless.ino tabs work unchanged (rad where noted)
float p_legFreq  = 1.56f;                     // [Hz]
float p_legAmp   =  85.0f * PI / 180.0f;      // [rad] crank amplitude (unipolar)
float p_hipAmp   = 28.0f  * PI / 180.0f;      // [rad] hip half-rectified swing
float p_hipPhiD  = 270.0f;                    // [deg] hip-vs-leg phase offset
float p_hipOffD  = 30.0f;                     // [deg] symmetric forward-pitch offset

// torso kappa-PID (sim TorsoKappaPID values)
float p_kappa = 2.0f, p_kp = 0.5f, p_ki = 0.1f;   // press '0' or '2' to switch kappa
// kp is adjustable at runtime ('4'..'7') because it is the one knob that decides whether
// the discrete loop is stable. Closing the loop once per control period, the joint error
// evolves as  dJ_{n+1} = dJ_n * [1 - g*(1+kp)]  where g is the fraction of the remaining
// gap the servo covers in one period. With PROFILE_VELOCITY=0 and a ~75 ms period the
// servo arrives within one step (g ~ 1), so the factor is just -kp: stable only for
// kp < 1, and kp=2 doubles the error every step with a sign flip -- which is exactly what
// the 2026-08-28 walk recorded (50% of consecutive samples reversed sign, command railed
// to the +-25 clamp 38% of the time, torso motor tripped on overload).
// The equilibrium J* = (kappa-1)*A does NOT depend on kp, so lowering kp changes how fast
// the torso converges, not what it converges to.
const float KP_TIERS[4] = {2.0f, 1.0f, 0.5f, 0.25f};   // keys '4' '5' '6' '7'

// The LEG motors' own internal position gain (control table addr 84, RAM, so writable
// with torque on). XM430 ships at 800. During the walk the cranks delivered only 0.816 of
// the commanded 95 deg -- worse than the 0.909 measured on the bench, where only two
// motors moved and nothing carried body weight. A stiffer internal loop tracks a 1.67 Hz
// sinusoid with less amplitude loss, and it moves the hardware TOWARD sim, where the crank
// follows its command at 0.99.
//
// The TORSO (ID 0) is deliberately left at the default. Its servo already closes ~2/3 of
// the gap per outer-loop period (g ~= 0.67); stiffening it pushes g toward 1 and eats the
// stability margin of the kappa loop, which needs g*(1+kp) < 2.
const uint16_t PGAIN_TIERS[4] = {800, 1200, 1600, 2000};   // key 'g' cycles these
int      pgain_idx = 2;      // boot at 1600: measured strictly better than the 800 default
uint16_t leg_pgain = 1600;
                                                  // (identical legs both ways -- that is the point)
const float TORSO_CLAMP_DEG = 25.0f;   // measured mechanical stop: +30.9/-31.9 at hip_off=0,
                                       // +27.8/-31.0 at hip_off=20 (torso_rom, 2026-08-28).
                                       // 25 keeps ~3 deg of margin off the nearest stop.
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
float   home_deg[MOTOR_COUNT];
unsigned long walk_start_ms = 0, autowait_ms = 0;
bool    wifi_active = false;
float torso_iErr = 0.0f;
float loop_dt_ema = 0.0f;      // measured control period [s], EMA
unsigned long loop_count = 0;  // loops since the last 1 Hz report (true rate, telemetry or not)
bool  stream_dbg = false;          // 't' toggles a 5 Hz roll/joint/axis stream (hand-rock test)

const float READY_STEP = 1.0f, READY_STEP_TORSO = 1.5f, ARRIVE_THRESH = 1.0f;
const float HIP_REST_DEG = 10.0f;   // rest lean: hips 10 deg forward or the robot tips backward

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
float zeroExtended(float cur) {       // extended-coord value of physical 0 nearest to cur
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  return cur - ((phys > 180.0f) ? phys - 360.0f : phys);
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
    float boot_deg = dxl.getPresentPosition(id, UNIT_DEGREE);
    home_deg[i] = zeroExtended(boot_deg);   // absolute zero, in this motor's extended coords
    dxl.torqueOff(id);
    // EXTENDED position: accepts negative / >360 deg goals. In plain OP_POSITION a goal
    // like home-20 with home near 0 deg is silently REJECTED (suspected hip/torso no-move).
    dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
    // unlimited profile so 1.67 Hz targets are tracked, not slew-limited
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.torqueOn(id);
    dxl.setGoalPosition(id, boot_deg, UNIT_DEGREE);   // hold boot pose (no snap); READY walks to zero
    DEBUG_SERIAL.print("ID "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print(" boot = "); DEBUG_SERIAL.print(boot_deg, 2);
    DEBUG_SERIAL.println("  (home = absolute 0)");
  }

  if (!bno.begin()) { DEBUG_SERIAL.println("ERROR: BNO055 not detected."); while (1); }
  bno.setExtCrystalUse(true);


  begin_wifi();                                  // webpage + /cmd + /data (wireless.ino tab)


  DEBUG_SERIAL.println("pengu_champ ready. r=Ready w=Walk q=Idle s=SignCheck (serial only)");
  if (AUTO_WALK) { robot_state = STATE_READY_SLIDE; DEBUG_SERIAL.println("AUTO: -> READY"); }
}

// The control law needs exactly one number from the IMU: imu_roll. The linear-accel
// vector and the calibration bytes are telemetry, and each is a separate I2C transaction
// on the critical path, so they are read at 1 Hz instead of every control step.
void update_imu() {
  imu::Vector<3> e = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  imu_yaw = e.x(); imu_roll = e.y(); imu_pitch = e.z();   // same mapping as pengu.ino

  static unsigned long slow = 0;
  if (millis() - slow > 1000) {
    slow = millis();
    imu::Vector<3> a = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
    bno.getCalibration(&cal_sys, &cal_gyro, &cal_accel, &cal_mag);
    imu_ax = a.x(); imu_ay = a.y(); imu_az = a.z();
  }
}

// ===================== WALK: c6 champion =====================
// legs and hips only -- never the torso, see PGAIN_TIERS above
void setLegPGain(uint16_t v) {
  const uint8_t LEGS[4] = {XM_RIGHT_HIP, XM_LEFT_HIP, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};
  for (int i = 0; i < 4; i++) dxl.writeControlTableItem(POSITION_P_GAIN, LEGS[i], v);
  leg_pgain = v;
}

void run_walk() {
  float t = (millis() - walk_start_ms) / 1000.0f;

  // staged start: ramp hip_off -> settle -> blend the oscillation in
  float off_deg = HIP_REST_DEG + (p_hipOffD - HIP_REST_DEG) * constrain(t / T_RAMP, 0.0f, 1.0f);
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
  // integrate with the MEASURED period: the loop is ~75 ms, not the 20 ms the hardcoded
  // 0.02 assumed, so the integral used to accumulate 3.7x slower than designed
  static unsigned long ctrl_prev_ms = 0;
  unsigned long ctrl_now_ms = millis();
  float ctrl_dt = ctrl_prev_ms ? (ctrl_now_ms - ctrl_prev_ms) * 0.001f : 0.02f;
  ctrl_prev_ms = ctrl_now_ms;
  if (ctrl_dt < 0.001f || ctrl_dt > 0.5f) ctrl_dt = 0.02f;      // ignore pauses/first call
  loop_dt_ema = (loop_dt_ema <= 0.0f) ? ctrl_dt : (0.9f * loop_dt_ema + 0.1f * ctrl_dt);
  torso_iErr   = constrain(torso_iErr + err * ctrl_dt, -20.0f, 20.0f);
  float torso_deg = alpha * constrain(
      (p_kappa - 1.0f) * axis / S_TILT + p_kp * err + p_ki * torso_iErr,
      -TORSO_CLAMP_DEG, TORSO_CLAMP_DEG);

  // sign conventions inherited from pengu.ino (L negative / R positive); see check #2
  dxl.setGoalPosition(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)]  - magL,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)] + magR,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_HIP,    home_deg[idxOf(XM_LEFT_HIP)]    - hipL_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_HIP,   home_deg[idxOf(XM_RIGHT_HIP)]   + hipR_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_TORSO_ROLL,  home_deg[idxOf(XM_TORSO_ROLL)]  + torso_deg, UNIT_DEGREE);


  // ---- 20 Hz CSV telemetry (toggle with 't'); control logic above is untouched ----
  // Costs 5 extra reads on telemetry ticks only; dt_ms is logged so the perturbation is
  // visible rather than assumed. Compare against grid5/walk_prediction.py.
  static unsigned long tel = 0, telPrev = 0;
  if (stream_dbg && millis() - tel >= 50) {
    unsigned long nowMs = millis();
    float dtms = telPrev ? (float)(nowMs - telPrev) : 0.0f;
    tel = nowMs; telPrev = nowMs;
    float pSlL = dxl.getPresentPosition(XM_LEFT_SLIDE,  UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_SLIDE)];
    float pSlR = dxl.getPresentPosition(XM_RIGHT_SLIDE, UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_SLIDE)];
    float pHpL = dxl.getPresentPosition(XM_LEFT_HIP,    UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_HIP)];
    float pHpR = dxl.getPresentPosition(XM_RIGHT_HIP,   UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_HIP)];
    float mA   = (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, XM_TORSO_ROLL) * 2.69f;
    DEBUG_SERIAL.print("w,");    DEBUG_SERIAL.print(t, 3);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(alpha, 3);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(-magL, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(pSlL, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(magR, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(pSlR, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(-hipL_deg, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(pHpL, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(hipR_deg, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(pHpR, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(torso_deg, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(J_deg, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(mA, 0);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(imu_roll, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(imu_pitch, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(axis, 2);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.println(dtms, 1);
  }

  // 1 Hz debug: what is actually being commanded (catch out-of-range / alpha timing)
  static unsigned long dbg = 0;
  if (millis() - dbg > 1000) { dbg = millis();
    DEBUG_SERIAL.print("t=");      DEBUG_SERIAL.print(t, 1);
    DEBUG_SERIAL.print(" a=");     DEBUG_SERIAL.print(alpha, 2);
    DEBUG_SERIAL.print(" hipL->"); DEBUG_SERIAL.print(home_deg[idxOf(XM_LEFT_HIP)] - hipL_deg, 1);
    DEBUG_SERIAL.print(" hipR->"); DEBUG_SERIAL.print(home_deg[idxOf(XM_RIGHT_HIP)] + hipR_deg, 1);
    DEBUG_SERIAL.print(" torso->");DEBUG_SERIAL.print(home_deg[idxOf(XM_TORSO_ROLL)] + torso_deg, 1);
    DEBUG_SERIAL.print(" roll=");  DEBUG_SERIAL.print(imu_roll, 1);
    DEBUG_SERIAL.print(" axis=");  DEBUG_SERIAL.print(axis, 1);
    DEBUG_SERIAL.print(" | kp=");  DEBUG_SERIAL.print(p_kp, 2);
    DEBUG_SERIAL.print(" k=");     DEBUG_SERIAL.print(p_kappa, 1);
    DEBUG_SERIAL.print(" loop=");  DEBUG_SERIAL.print(1000.0f * loop_dt_ema, 1);
    DEBUG_SERIAL.print("ms (");    DEBUG_SERIAL.print(loop_dt_ema > 0 ? 1.0f / loop_dt_ema : 0.0f, 1);
    DEBUG_SERIAL.print(" Hz)  loops/s=");  DEBUG_SERIAL.print(loop_count);
    loop_count = 0;
    DEBUG_SERIAL.print("  step gain (g=1) = "); DEBUG_SERIAL.print(p_kp, 2);
    DEBUG_SERIAL.print(" | legP=");  DEBUG_SERIAL.print(leg_pgain);
    DEBUG_SERIAL.print(" V=");
    DEBUG_SERIAL.print(dxl.readControlTableItem(PRESENT_INPUT_VOLTAGE, XM_LEFT_SLIDE) * 0.1f, 1);
    int tmax = 0;
    const uint8_t ALLM[5] = {XM_TORSO_ROLL, XM_RIGHT_HIP, XM_LEFT_HIP, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};
    for (int i = 0; i < 5; i++) {
      int tc = dxl.readControlTableItem(PRESENT_TEMPERATURE, ALLM[i]);
      if (tc > tmax) tmax = tc;
    }
    DEBUG_SERIAL.print("V Tmax="); DEBUG_SERIAL.print(tmax); DEBUG_SERIAL.println("C");
  }
}

void loop() {
  update_imu();

  char cmd = 0;
  if (wifi_active) cmd = update_wifi();          // /cmd?key=X from the webpage
  if (DEBUG_SERIAL.available()) cmd = (char)DEBUG_SERIAL.read();   // serial wins
  switch (cmd) {
    case 'r': robot_state = STATE_READY_SLIDE; DEBUG_SERIAL.println("-> READY"); break;
    case 'w':
      if (robot_state == STATE_IDLE) {
        walk_start_ms = millis(); torso_iErr = 0;
        robot_state = STATE_WALK; DEBUG_SERIAL.println("-> WALK (staged start)");
      } else DEBUG_SERIAL.println("Go READY first.");
      break;
    case 'q': robot_state = STATE_IDLE; DEBUG_SERIAL.println("-> IDLE"); break;
    case 'g': {
      pgain_idx = (pgain_idx + 1) % 4;
      setLegPGain(PGAIN_TIERS[pgain_idx]);
      DEBUG_SERIAL.print("# leg POSITION_P_GAIN = "); DEBUG_SERIAL.print(leg_pgain);
      DEBUG_SERIAL.println("   (hips+cranks only; torso left at default 800)");
      break;
    }
    case '4': case '5': case '6': case '7': {
      p_kp = KP_TIERS[cmd - '4'];
      torso_iErr = 0;
      DEBUG_SERIAL.print("# kp = "); DEBUG_SERIAL.print(p_kp, 2);
      DEBUG_SERIAL.print("   -> per-step error gain (g=1) = "); DEBUG_SERIAL.print(p_kp, 2);
      DEBUG_SERIAL.println(p_kp < 1.0f ? "  STABLE (converges)" : "  UNSTABLE (diverges, sign-flips)");
      break;
    }
    case 'n': case 'm': case 'j': {   // switch gait without reflashing
      if (cmd == 'n') {        // lean 30: cranks fit AND the torso fits, with margin
        p_legFreq = 1.56f; p_legAmp =  85.0f*PI/180.0f; p_hipAmp = 28.0f*PI/180.0f;
        p_hipPhiD = 270.0f; p_hipOffD = 30.0f;
      } else if (cmd == 'm') { // lean 40, if 30 still sits back; torso command clips
        p_legFreq = 1.42f; p_legAmp =  85.0f*PI/180.0f; p_hipAmp = 28.0f*PI/180.0f;
        p_hipPhiD = 330.0f; p_hipOffD = 40.0f;
      } else {                 // 'j': lean 50, the most forward; torso clips hardest
        p_legFreq = 1.36f; p_legAmp =  95.0f*PI/180.0f; p_hipAmp = 28.0f*PI/180.0f;
        p_hipPhiD = 260.0f; p_hipOffD = 50.0f;
      }
      torso_iErr = 0;
      DEBUG_SERIAL.print("# gait -> "); DEBUG_SERIAL.print(p_legFreq, 2);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_hipPhiD, 0);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_legAmp * 180.0f / PI, 0);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_hipAmp * 180.0f / PI, 0);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_hipOffD, 0);
      DEBUG_SERIAL.print("   crank demand "); DEBUG_SERIAL.print(PI * p_legFreq * p_legAmp * 180.0f / PI, 0);
      DEBUG_SERIAL.print(" deg/s, sim torso cmd +-");
      DEBUG_SERIAL.println(cmd == 'n' ? "14.8 deg (fits)" : (cmd == 'm' ? "33.5 deg (clips)" : "35.3 deg (clips)"));
      break;
    }
    case '0': case '2': {                 // A/B the torso law without touching anything else
      p_kappa = (cmd == '0') ? 0.0f : 2.0f;
      torso_iErr = 0;
      DEBUG_SERIAL.print("# kappa = "); DEBUG_SERIAL.print(p_kappa, 1);
      DEBUG_SERIAL.print("  (sim predicts torso world roll rms ");
      DEBUG_SERIAL.print(p_kappa > 1.0f ? "14.5" : "-");
      DEBUG_SERIAL.println(" deg at mu=0.5, COM 1.05, kappa=2)");
      break;
    }
    case 's': {
      float J = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - home_deg[idxOf(XM_TORSO_ROLL)];
      DEBUG_SERIAL.print("imu_roll="); DEBUG_SERIAL.print(imu_roll, 2);
      DEBUG_SERIAL.print("  torso_joint="); DEBUG_SERIAL.print(J, 2);
      DEBUG_SERIAL.println("  (rotate torso by hand: same direction -> S_TILT=+1, opposite -> -1)");
      break;
    }
    case 't': stream_dbg = !stream_dbg;
      if (stream_dbg) {
  DEBUG_SERIAL.println("w,t,alpha,goal_slL,pos_slL,goal_slR,pos_slR,goal_hipL,pos_hipL,goal_hipR,pos_hipR,goal_torso,pos_torso,mA_torso,imu_roll,imu_pitch,axis,dt_ms");
      } else DEBUG_SERIAL.println("stream OFF");
      break;
    case 'p': {                                    // motor health report
      for (int i = 0; i < MOTOR_COUNT; i++) {
        uint8_t id = MOTOR_IDS[i];
        bool ok = dxl.ping(id);
        float pos = dxl.getPresentPosition(id, UNIT_DEGREE);
        int32_t hwerr = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, id);
        int32_t tq    = dxl.readControlTableItem(TORQUE_ENABLE, id);
        DEBUG_SERIAL.print("ID "); DEBUG_SERIAL.print(id);
        DEBUG_SERIAL.print(ok ? "  ping OK" : "  PING FAIL");
        DEBUG_SERIAL.print("  pos="); DEBUG_SERIAL.print(pos, 1);
        DEBUG_SERIAL.print("  torque="); DEBUG_SERIAL.print(tq);
        DEBUG_SERIAL.print("  hwErr="); DEBUG_SERIAL.println(hwerr);   // 0 = healthy
      }
      break;
    }
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
      stepMotorToward(XM_LEFT_SLIDE,  0.0f, READY_STEP);
      stepMotorToward(XM_RIGHT_SLIDE, 0.0f, READY_STEP);
      if (arrivedAt(XM_LEFT_SLIDE, 0.0f) && arrivedAt(XM_RIGHT_SLIDE, 0.0f)) {
        robot_state = STATE_READY_HIP; DEBUG_SERIAL.println("Slides at zero -> hips (rest lean)");
      }
      break;
    }
    case STATE_READY_HIP: {
      stepMotorToward(XM_LEFT_HIP,  360.0f - HIP_REST_DEG, READY_STEP);   // L: -10 deg (fwd)
      stepMotorToward(XM_RIGHT_HIP,  HIP_REST_DEG,          READY_STEP);   // R: +10 deg (fwd)
      if (arrivedAt(XM_LEFT_HIP, 360.0f - HIP_REST_DEG) && arrivedAt(XM_RIGHT_HIP, HIP_REST_DEG)) {
        robot_state = STATE_READY_TORSO; DEBUG_SERIAL.println("Hips at rest lean -> torso");
      }
      break;
    }
    case STATE_READY_TORSO:
      stepMotorToward(XM_TORSO_ROLL, 0.0f, READY_STEP_TORSO);
      if (arrivedAt(XM_TORSO_ROLL, 0.0f)) {
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

  // No pacing delay. The inherited delay(20) was meant to hold ~50 Hz, but the measured
  // period was 37-42 ms (25 Hz) because the delay sat ON TOP of the real work. It bought
  // nothing: the gait phase is computed from absolute time (millis()), so the waveform is
  // identical either way, and the integral now uses the measured period rather than a
  // hardcoded 0.02. Removing it raises the control rate, which lowers g (the fraction of
  // the gap the servo closes per period) and so raises the kp the loop can carry --
  // stability needs g*(1+kp) < 2, and sim runs kp=2 at 1 kHz.
  loop_count++;
}
