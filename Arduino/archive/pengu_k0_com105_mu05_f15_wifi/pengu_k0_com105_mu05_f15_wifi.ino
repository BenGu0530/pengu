// pengu_k0_com105_mu05_f15_wifi.ino — WiFi build; every command has a button at
// 192.168.4.1.
//
// ---------------------------------------------------------------------------------------
// pengu_k0_com105_mu05_f15.ino — GRID-4 config c1 (kappa=0), COM 1.05, mu = 0.5,
// gaits taken from the freq 1.4-1.6 band.
//
// >>> HARDWARE CHANGE REQUIRED <<<  COM 1.05 = counterweight slid DOWN ~86 mm along the
// torso mast from the 1.31 build (sim slide -86.05 mm, mass 2.2724 kg).
//
// All three lean 30 deg (the earlier set sat backwards at 10-20) and all three come from
// the same freq band, so the only thing that really changes across n -> m -> j is how far
// the crank demand exceeds what the motors can turn. That makes this set a direct test of
// where the velocity envelope starts to bite:
//
//        gait                     sweep net_fwd   nbhd   crank demand   vs ~430 ceiling
//   'n'  1.43 / 260 /  95 / 28 / 30   0.1108      1.00      427 deg/s   inside
//   'm'  1.52 / 250 / 105 / 28 / 30   0.1282      0.78      501 deg/s   17% over
//   'j'  1.59 / 280 / 125 / 20 / 30   0.1383      0.78      624 deg/s   45% over
//
// The sweep ranks them j > m > n. If the envelope matters, the hardware should rank them
// the other way round. Neighbourhood score is 1.00 for 'n', meaning every one of its
// adjacent cells also walks, against 0.78 for the other two.
//
// Nothing is off-grid here: all five parameters of each gait are exactly the cells the
// sweep selected. Configuration stays c1, kappa stays 0.
//
// Sim reference, mu=0.5, kappa=0, COM 1.05 (grid5 staged protocol, 10 s window):
//
//                        n (1.43)   m (1.52)   j (1.59)
//   speed                0.112      0.125      0.144  m/s
//   torso world roll     1.8        1.7        2.6    deg rms   (kappa=0 holds it level)
//   torso command amp    42.6       34.4       49.3   deg       (clamped to +-25)
//   torso torque p95     0.65       0.58       0.41   N.m       (stall 4.1, far from it)
//   crank torque p95     0.83       0.98       1.75   N.m
//
// The torso command still exceeds the +-25 clamp on all three, so it will be clipped --
// kappa=0 asks for a large torso swing on this body. Torque is never a problem here.
//
// Everything else inherited: kp = 0.5, no pacing delay (57-80 Hz), integral on the
// measured period, leg POSITION_P_GAIN booting at 1600.
//
// Commands:  r = READY   w = WALK   q = IDLE   0 / 2 = kappa   n / m / j = gait
//            t = telemetry   e = 20/50 Hz   4 / 5 / 6 / 7 = kp   g = leg P gain
//            p = health     i = re-init motors (torque back on after a late power-up)
//            0 / 1 / 2 = kappa (1 = fixed joint, the control condition)
//            c = torso torque limit  3210/800/600/400/250 mA
//            v = torso angle  limit  25/20/15/10/5 deg
//            d = torso kd           0/0.02/0.05/0.10/0.20 s
//            SINGLE MECHANISM:  x = hip swing only   y = leg extension only
//                              u = torso only (open loop)   , / . = freq -/+ 0.1 Hz
//            Over WiFi every one of these has a button on the page.

// ======================= SET THESE BEFORE UPLOADING =======================
// These are not live controls. They are chosen once, flashed, and left alone; the serial
// keys that change them still exist for bench work but they are off the web page, because
// a value that has to be re-picked after every power cycle is a value that will be wrong
// in half the records.
//
//   kappa            0    torso held level in the world (config c1)
//   kp               0.5  measured optimum; 1.0 is marginal and 2.0 diverges on this robot
//   ki               0.1  contributes under 1 deg; effectively inert
//   kd               0    D on the torso loop, in seconds. 0.05 matches the whole kp term
//   torso clamp     25    deg. Mechanical stop measured +27.8 / -31.0
//   torso current 3210    mA. Register maximum, i.e. no cap. Under 830 to actually bite
//   leg P gain    1600    servo POSITION_P_GAIN on hips and cranks; torso left at default
//   telemetry       on    CSV streams from the moment WALK starts
// =========================================================================
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
float p_legFreq  = 1.43f;                     // [Hz]
float p_legAmp   =  95.0f * PI / 180.0f;      // [rad] crank amplitude (unipolar)
float p_hipAmp   = 28.0f  * PI / 180.0f;      // [rad] hip half-rectified swing
float p_hipPhiD  = 260.0f;                    // [deg] hip-vs-leg phase offset
float p_hipOffD  = 30.0f;                     // [deg] symmetric forward-pitch offset

// Changing the frequency mid-walk is not free. The phase was computed as
// 2*pi*f*(t - t_start), so raising f from 1.4 to 1.5 at t = 13 s past the start jumps the
// phase by 2*pi*0.1*13 = 1.3 cycles -- a 0.3-cycle STEP in the hip command, which kicks the
// robot every time the frequency is touched. setFreq() absorbs that into an offset so the
// waveform is continuous across the change; at a constant frequency it does nothing.
float gait_phi_off = 0.0f;

// SINGLE-MECHANISM MODES. Ben, 2026-08-30: this robot walks on any ONE of its three
// mechanisms alone -- torso alone, leg extension alone, or leg swing alone. Simulation
// does not agree: with COM 1.05 at mu 0.5 it walks on hip swing alone (net 0.004 -> 0.081
// m/s from 0.6 to 2.0 Hz) but goes essentially nowhere on leg extension alone (net <=
// 0.018 despite moving about) and nowhere at all on torso alone, which simply falls over
// above 1.4 Hz. Which of the three transfer, and at what frequency, is a binary readout --
// it walked or it did not -- with no axis convention, phase definition, or fit quality in
// the way. It is the cleanest test available of where the model is wrong.
//
// The torso mode has to bypass the kappa loop: with the hips and cranks still, the hip
// axis does not roll, so the loop would correctly command nothing. This drives the torso
// joint open-loop instead, which is what the simulation's torso_amp does.
bool  torso_open_loop = false;
float torso_ol_amp_deg = 25.0f;

// torso kappa-PID (sim TorsoKappaPID values)
float p_kappa = 0.0f, p_kp = 0.5f, p_ki = 0.1f;   // press '0' or '2' to switch kappa
// DERIVATIVE on the torso loop. Measured on gnd_n_k0 (clean walking cycles): the torso's
// world roll rate is rms 88.8 deg/s, p95 196, against a roll of only ~9 deg rms -- the rate
// is a factor of ten larger than the angle, which is why kd has more authority here than kp.
// kd = 0.05 s already contributes as much as the whole kp term (4.4 deg rms each).
//
// What it does and does not do: with kappa=0 the error is -roll, so the D term is
// -kd*d(roll)/dt, i.e. it opposes whatever rotation the torso is doing right now. That is
// real damping of the torso's world roll. It does NOT make the loop ask for less -- it is
// one more term added to a command already dominated by the (kappa-1)*axis feedforward, so
// expect MORE clamp hits, not fewer. Damping the rate and capping the amplitude are two
// different levers; this is the rate one.
//
// Caveat the data cannot settle: only 23.8 of that 88.8 deg/s is at the gait frequency. The
// rest is the non-repeating motion, and D will fight that too -- whether it damps it or
// chases it only a run can say.
//
// The rate comes from differencing imu_roll, NOT from the gyro. Differencing is noisier
// (the 0.07 deg quantisation gives 4.7 deg/s against an 88.8 signal, so 19:1) but its SIGN
// is right by construction, because it is the derivative of the very quantity in the error.
// The gyro would need its axis identified first, and a sign error there is positive
// feedback. Its three axes are logged so that can be settled from data, then swapped in.
float TORSO_KD = 0.0f;                        // 'd' steps 0 / 0.02 / 0.05 / 0.10 / 0.20
const float KD_STEPS[5] = {0.0f, 0.02f, 0.05f, 0.10f, 0.20f};
int kd_idx = 0;
float roll_rate = 0.0f;                       // [deg/s] EMA of d(imu_roll)/dt, tau ~30 ms
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

// ===================== The two limits on the torso =====================
// Gait, kappa, kp, ki are all unchanged. These cap what the torso is ALLOWED to do.
//
// Measured on gnd_n_k0 (2026-08-30, gait n, kappa=0, clean walking cycles only):
//   torso joint swings 32.7 deg peak-to-peak, i.e. +-16, against the +-25 clamp
//   the command sits ON the clamp 17% of the time
//   torso current p95 830 mA, peak 1829 mA
//   the lower body's roll is 37.1 deg p2p with the loop pushing, 31.6 with it idle
//
// Why these two and not the PID gains: the command is (kappa-1)*axis + kp*err + ki*int,
// and on that data the feedforward term is rms 17.8 against the feedback's 4.4. Taking kp
// from 0.5 to 0 only moves the clamp hits from 19% to 15%, and the integral contributes
// under 1 deg. The gains have almost no authority here; these two limits do.
//
// ANGLE limit -- how far the loop may swing the torso. Mechanical stop measured at
// +27.8 / -31.0 deg (torso_rom, 2026-08-28), so 25 was margin, not a control choice.
float TORSO_CLAMP_DEG = 25.0f;                       // 'v' steps 25 / 20 / 15 / 10 / 5
const float CLAMP_STEPS[5] = {25.0f, 20.0f, 15.0f, 10.0f, 5.0f};
int clamp_idx = 0;

// TORQUE limit -- how hard the motor may push, in mA. Written to GOAL_CURRENT, which needs
// the torso in OP_CURRENT_BASED_POSITION (still multi-turn, so the position goals below are
// unchanged). The XM430 register tops out at 3210 mA and the motor's own stall at 12 V is
// about 1300, so 3210 is "no cap" -- it reproduces the runs already recorded. To bite it
// has to sit under the measured 830 mA p95.
uint16_t TORSO_CURRENT_MA = 3210;                    // 'c' steps 3210 / 800 / 600 / 400 / 250
const uint16_t CURRENT_STEPS[5] = {3210, 800, 600, 400, 250};
int current_idx = 0;
int      pgain_idx = 2;      // boot at 1600: measured strictly better than the 800 default
uint16_t leg_pgain = 1600;
                                                  // (identical legs both ways -- that is the point)
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
float   imu_gx = 0, imu_gy = 0, imu_gz = 0;   // logged only; not in the loop yet
uint8_t cal_sys = 0, cal_gyro = 0, cal_accel = 0, cal_mag = 0;
float   home_deg[MOTOR_COUNT];
unsigned long walk_start_ms = 0, autowait_ms = 0;
bool    wifi_active = false;
float torso_iErr = 0.0f;
float loop_dt_ema = 0.0f;      // measured control period [s], EMA
unsigned long loop_count = 0;  // loops since the last 1 Hz report (true rate, telemetry or not)
// The CSV stream is ON by default. It was off, and on 2026-08-30 19:25 a whole 36 s walk
// came back with 37 one-second debug lines and no data because 't' was not pressed. The
// rows only print inside run_walk(), so leaving it on costs nothing until the robot walks.
bool  stream_dbg = true;       // 't' toggles the CSV stream
// Telemetry period. 20 Hz (50 ms) is the default because it costs 5 extra servo reads on
// every telemetry tick and those sit inside the control loop. At 20 Hz a 1.43 Hz gait gets
// 14 samples per cycle -- plenty to fit the fundamental, but useless above ~7 Hz, and a
// feedback loop with the measured 74 ms of delay would oscillate right around 7-8 Hz.
// Key 'e' switches to 50 Hz (25 Hz Nyquist) for one diagnostic run; dt_ms is logged, so
// what the extra reads cost the control period is visible in the data rather than assumed.
uint16_t tel_ms = 50;          // 'e' toggles 50 ms (20 Hz) <-> 20 ms (50 Hz)

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

// Bringing the motors up. Split out of setup() because the board is powered by USB and
// the motors are not: opening the serial port resets the board (DTR), so if the 12 V comes
// on afterwards, setup() has already run against dead motors and every torqueOn was lost.
// That is exactly what 2026-08-30 19:15 recorded -- all five reporting ping OK, torque=0,
// no boot banner in the log, READY printing but nothing moving. Press 'i' to redo it.
bool initMotors(bool verbose) {
  bool all_ok = true;
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(2.0);
  for (int i = 0; i < MOTOR_COUNT; i++) {
    uint8_t id = MOTOR_IDS[i];
    if (!dxl.ping(id)) {
      all_ok = false;
      DEBUG_SERIAL.print("No response ID "); DEBUG_SERIAL.print(id);
      DEBUG_SERIAL.println("   <-- is the 12 V on?");
      continue;
    }
    float boot_deg = dxl.getPresentPosition(id, UNIT_DEGREE);
    home_deg[i] = zeroExtended(boot_deg);   // absolute zero, in this motor's extended coords
    dxl.torqueOff(id);
    // EXTENDED position: accepts negative / >360 deg goals. In plain OP_POSITION a goal
    // like home-20 with home near 0 deg is silently REJECTED (suspected hip/torso no-move).
    // The torso runs current-based position so its torque ceiling is settable ('c');
    // mode 5 is still multi-turn, so the extended-position goals are unchanged.
    dxl.setOperatingMode(id, (id == XM_TORSO_ROLL) ? OP_CURRENT_BASED_POSITION
                                                   : OP_EXTENDED_POSITION);
    // unlimited profile so the gait's targets are tracked, not slew-limited
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.torqueOn(id);
    if (id == XM_TORSO_ROLL) setTorsoCurrent(TORSO_CURRENT_MA);
    dxl.setGoalPosition(id, boot_deg, UNIT_DEGREE);   // hold where it is; no snap
    // read it back rather than assume: a torqueOn against an unpowered bus reports nothing
    int32_t tq = dxl.readControlTableItem(TORQUE_ENABLE, id);
    if (tq != 1) all_ok = false;
    if (verbose) {
      DEBUG_SERIAL.print("ID "); DEBUG_SERIAL.print(id);
      DEBUG_SERIAL.print(" boot = "); DEBUG_SERIAL.print(boot_deg, 2);
      DEBUG_SERIAL.print("  torque="); DEBUG_SERIAL.println(tq);
    }
  }
  DEBUG_SERIAL.println(all_ok ? "# all 5 motors powered and holding"
                              : "# NOT ALL MOTORS ARE HOLDING -- power the 12 V, then press 'i'");
  return all_ok;
}

void setup() {
  DEBUG_SERIAL.begin(115200);
  unsigned long t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 3000);   // don't block untethered boots

  initMotors(true);

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
  imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu_gx = g.x(); imu_gy = g.y(); imu_gz = g.z();          // deg/s, sensor frame
  // roll rate by differencing, filtered lightly. tau ~30 ms is far below the 700 ms gait
  // period, so it costs no phase where it matters, and it keeps the quantisation steps from
  // showing up as spikes in the D term.
  static unsigned long r_prev_ms = 0;
  static float roll_prev = 0.0f;
  unsigned long r_now = millis();
  if (r_prev_ms) {
    float rdt = (r_now - r_prev_ms) * 0.001f;
    if (rdt > 0.001f && rdt < 0.2f) {
      float raw = (imu_roll - roll_prev) / rdt;
      float w = rdt / (0.03f + rdt);
      roll_rate = (1.0f - w) * roll_rate + w * raw;
    }
  }
  r_prev_ms = r_now; roll_prev = imu_roll;

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
void setTorsoCurrent(uint16_t mA) {
  TORSO_CURRENT_MA = mA;
  dxl.writeControlTableItem(GOAL_CURRENT, XM_TORSO_ROLL, (uint16_t)(mA / 2.69f));
}

void setFreq(float f) {
  if (robot_state == STATE_WALK) {
    float tw = (millis() - walk_start_ms) / 1000.0f - T_RAMP - T_SETTLE;
    if (tw > 0.0f) gait_phi_off += 2.0f * PI * (p_legFreq - f) * tw;   // bumpless
  }
  p_legFreq = f;
}

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
  float phase   = (t > T_RAMP + T_SETTLE)
                  ? 2.0f * PI * p_legFreq * (t - T_RAMP - T_SETTLE) + gait_phi_off : 0.0f;

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
  // d(err)/dt = d(kappa*axis - roll)/dt; with kappa=0 that is exactly -roll_rate. The axis
  // term is left out of the derivative on purpose: axis = roll - J carries the torso motor's
  // own motion, so differentiating it would feed the actuator back into its own D term.
  float dErr = -roll_rate;
  float torso_deg;
  if (torso_open_loop) {
    torso_iErr = 0.0f;                         // the loop is not running; do not wind up
    torso_deg  = alpha * torso_ol_amp_deg * sinf(phase);
  } else {
    torso_deg = alpha * constrain(
        (p_kappa - 1.0f) * axis / S_TILT + p_kp * err + p_ki * torso_iErr + TORSO_KD * dErr,
        -TORSO_CLAMP_DEG, TORSO_CLAMP_DEG);
  }

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
  if (stream_dbg && millis() - tel >= tel_ms) {
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
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(dtms, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(TORSO_CURRENT_MA);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(TORSO_CLAMP_DEG, 0);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(roll_rate, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(TORSO_KD, 3);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(imu_gx, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(imu_gy, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(imu_gz, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(p_legFreq, 3);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(p_legAmp * 180.0f / PI, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.println(p_hipAmp * 180.0f / PI, 1);
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
        walk_start_ms = millis(); torso_iErr = 0; gait_phi_off = 0.0f;
        robot_state = STATE_WALK; DEBUG_SERIAL.println("-> WALK (staged start)");
        // header per bout, so a capture holding several walks is still parsable
        if (stream_dbg) DEBUG_SERIAL.println("w,t,alpha,goal_slL,pos_slL,goal_slR,pos_slR,goal_hipL,pos_hipL,goal_hipR,pos_hipR,goal_torso,pos_torso,mA_torso,imu_roll,imu_pitch,axis,dt_ms,i_lim_mA,clamp_deg,roll_rate,kd,gx,gy,gz,freq,leg_amp,hip_amp");
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
      if (cmd == 'n') {        // the champion of the whole c1 mu=0.5 grid
        setFreq(1.95f); p_legAmp = 125.0f*PI/180.0f; p_hipAmp = 28.0f*PI/180.0f;
        p_hipPhiD =   0.0f; p_hipOffD = 20.0f;
      } else if (cmd == 'm') { // the most robust cell of that region (neighbourhood 1.00)
        setFreq(2.00f); p_legAmp = 115.0f*PI/180.0f; p_hipAmp = 28.0f*PI/180.0f;
        p_hipPhiD =   0.0f; p_hipOffD = 20.0f;
      } else {                 // same region, lowest crank demand that still runs fast
        setFreq(1.69f); p_legAmp = 125.0f*PI/180.0f; p_hipAmp = 28.0f*PI/180.0f;
        p_hipPhiD = 350.0f; p_hipOffD = 20.0f;
      }
      torso_iErr = 0;
      DEBUG_SERIAL.print("# gait -> "); DEBUG_SERIAL.print(p_legFreq, 2);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_hipPhiD, 0);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_legAmp * 180.0f / PI, 0);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_hipAmp * 180.0f / PI, 0);
      DEBUG_SERIAL.print(" / "); DEBUG_SERIAL.print(p_hipOffD, 0);
      DEBUG_SERIAL.print("   crank demand "); DEBUG_SERIAL.print(PI * p_legFreq * p_legAmp * 180.0f / PI, 0);
      DEBUG_SERIAL.print(" deg/s, sim torso cmd +-");
      DEBUG_SERIAL.println("clipped by the 25 deg clamp");
      break;
    }
    case 'x': {   // ROCK MODE: no leg extension, hips only -- the resonance probe
      // With leg_amp = 0 the cranks never move, so the robot only rocks on its feet while
      // the hips swing. hip_phi is the hip's phase RELATIVE TO THE CRANK, so with a still
      // crank it is a pure time shift and has no effect -- verified in sim, roll amplitude
      // 8.39 / 8.61 / 8.57 deg at hip_phi 240 / 0 / 90.
      //
      // Sim reference for this exact setup (COM 1.05, mu 0.5, hip_amp 20, hip_off 10),
      // torso roll amplitude against drive frequency, torso locked. Roll here is the tilt
      // about the torso hinge -- the quantity the controller nulls, not a world-frame Euler
      // angle. (An earlier version of this comment quoted a 0.9 Hz peak of 16.9 deg; that
      // was the FORE-AFT tilt, a different axis, and it is withdrawn.)
      //   0.6 Hz 3.9   0.8 Hz 4.2   1.0 Hz 4.3   1.2 Hz 5.8   1.4 Hz 7.4   1.6 Hz 8.9
      //   0.7 Hz 4.4   0.9 Hz 4.2   1.1 Hz 5.1   1.3 Hz 6.4   1.5 Hz 8.1   1.8 Hz 9.8
      // No resonance in this band: the response climbs monotonically. Purity 78-100%.
      setFreq(0.60f); p_legAmp = 0.0f; p_hipAmp = 20.0f*PI/180.0f; p_hipOffD = 10.0f;
      torso_open_loop = false; torso_iErr = 0;
      DEBUG_SERIAL.println("# HIP SWING ONLY  leg_amp=0 hip_amp=20 hip_off=10  freq 0.60 Hz");
      DEBUG_SERIAL.println("#   sim walks on this one: net 0.004 (0.6 Hz) -> 0.081 m/s (2.0 Hz)");
      DEBUG_SERIAL.println("#   ',' = freq -0.1   '.' = freq +0.1   (bumpless, safe mid-walk)");
      break;
    }
    case 'y': {   // LEG EXTENSION ONLY
      setFreq(0.60f); p_legAmp = 95.0f*PI/180.0f; p_hipAmp = 0.0f; p_hipOffD = 10.0f;
      torso_open_loop = false; torso_iErr = 0;
      DEBUG_SERIAL.println("# LEG EXTENSION ONLY  leg_amp=95 hip_amp=0 hip_off=10  freq 0.60 Hz");
      DEBUG_SERIAL.println("#   sim does NOT walk on this one: net <= 0.018 m/s at every freq");
      break;
    }
    case 'u': {   // TORSO ONLY, open loop (the kappa loop cannot drive a still body)
      setFreq(0.60f); p_legAmp = 0.0f; p_hipAmp = 0.0f; p_hipOffD = 10.0f;
      torso_open_loop = true; torso_iErr = 0;
      DEBUG_SERIAL.print("# TORSO ONLY  open-loop +-");
      DEBUG_SERIAL.print(torso_ol_amp_deg, 0);
      DEBUG_SERIAL.println(" deg, legs still, freq 0.60 Hz");
      DEBUG_SERIAL.println("#   sim does NOT walk on this one: nothing below 1.4 Hz, falls above");
      break;
    }
    case ',': case '.': {
      setFreq(constrain(p_legFreq + (cmd == '.' ? 0.10f : -0.10f), 0.30f, 2.60f));
      DEBUG_SERIAL.print("# freq = "); DEBUG_SERIAL.print(p_legFreq, 2);
      DEBUG_SERIAL.print(" Hz   crank demand ");
      DEBUG_SERIAL.print(PI * p_legFreq * p_legAmp * 180.0f / PI, 0);
      DEBUG_SERIAL.println(" deg/s");
      break;
    }
    case '0': case '1': case '2': {       // A/B the torso law without touching anything else
      // kappa = 1 is the FIXED-JOINT case: the target torso world roll equals the hip-axis
      // roll, so the error is just -J and the loop drives the torso joint to zero, i.e. the
      // torso rides with the axis and the torso motor stops fighting it. It is the control
      // condition for "is the broadband body motion coming from this loop or not" -- with
      // kappa=0 the command sat on the +-25 clamp 28% of the time (sim: 0%), and a clamped
      // loop is a nonlinearity that alone can scatter a single-frequency motion.
      p_kappa = (cmd == '0') ? 0.0f : (cmd == '1' ? 1.0f : 2.0f);
      torso_iErr = 0;
      DEBUG_SERIAL.print("# kappa = "); DEBUG_SERIAL.print(p_kappa, 1);
      DEBUG_SERIAL.println(p_kappa == 0.0f ? "  hold torso level in the world"
                         : (p_kappa == 1.0f ? "  FIXED JOINT: torso rides with the hip axis"
                                            : "  lean the torso into the roll"));
      break;
    }
    case 's': {
      float J = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - home_deg[idxOf(XM_TORSO_ROLL)];
      DEBUG_SERIAL.print("imu_roll="); DEBUG_SERIAL.print(imu_roll, 2);
      DEBUG_SERIAL.print("  torso_joint="); DEBUG_SERIAL.print(J, 2);
      DEBUG_SERIAL.println("  (rotate torso by hand: same direction -> S_TILT=+1, opposite -> -1)");
      break;
    }
    case 'c': {                                  // torque limit
      current_idx = (current_idx + 1) % 5;
      setTorsoCurrent(CURRENT_STEPS[current_idx]);
      DEBUG_SERIAL.print("# torso current limit = "); DEBUG_SERIAL.print(TORSO_CURRENT_MA);
      DEBUG_SERIAL.println(TORSO_CURRENT_MA > 830 ? " mA  (above the measured p95: no limit)"
                                                  : " mA  (below the measured 830 p95: bites)");
      break;
    }
    case 'd': {                                  // derivative gain
      kd_idx = (kd_idx + 1) % 5;
      TORSO_KD = KD_STEPS[kd_idx];
      torso_iErr = 0;
      DEBUG_SERIAL.print("# torso kd = "); DEBUG_SERIAL.print(TORSO_KD, 3);
      DEBUG_SERIAL.print(" s   -> contributes about ");
      DEBUG_SERIAL.print(TORSO_KD * 88.8f, 1);
      DEBUG_SERIAL.println(" deg rms (kp contributes 4.4)");
      break;
    }
    case 'v': {                                  // angle limit
      clamp_idx = (clamp_idx + 1) % 5;
      TORSO_CLAMP_DEG = CLAMP_STEPS[clamp_idx];
      torso_iErr = 0;
      DEBUG_SERIAL.print("# torso clamp = +-"); DEBUG_SERIAL.print(TORSO_CLAMP_DEG, 0);
      DEBUG_SERIAL.println(" deg   (the joint was swinging +-16 against +-25)");
      break;
    }
    case 'e': {
      tel_ms = (tel_ms == 50) ? 20 : 50;
      DEBUG_SERIAL.print("# telemetry ");  DEBUG_SERIAL.print(1000 / tel_ms);
      DEBUG_SERIAL.println(" Hz  (50 Hz resolves the 7-8 Hz band; 20 Hz cannot)");
      break;
    }
    case 't': stream_dbg = !stream_dbg;
      if (stream_dbg) {
  DEBUG_SERIAL.println("w,t,alpha,goal_slL,pos_slL,goal_slR,pos_slR,goal_hipL,pos_hipL,goal_hipR,pos_hipR,goal_torso,pos_torso,mA_torso,imu_roll,imu_pitch,axis,dt_ms,i_lim_mA,clamp_deg,roll_rate,kd,gx,gy,gz,freq,leg_amp,hip_amp");
      } else DEBUG_SERIAL.println("stream OFF");
      break;
    case 'i': {                                    // re-run the motor bring-up
      robot_state = STATE_IDLE;
      DEBUG_SERIAL.println("# re-initialising motors (use this if the 12 V came on after boot)");
      initMotors(true);
      break;
    }
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
  if (stream_dbg && robot_state != STATE_WALK && millis() - strm > 200) {
    strm = millis();
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
