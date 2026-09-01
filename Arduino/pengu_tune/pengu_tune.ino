// pengu_tune.ino — tune the gait by hand, over the serial terminal.
//
// Every gait taken from the sweep has failed on the robot. The clearest observation so far
// (2026-08-30): when the body rolls, the swing leg completes its stroke IN THE AIR and
// "finishes" a step that never loaded the ground. That is a timing failure between body
// roll and leg extension, not an amplitude one -- so the four things that decide timing and
// size are on keys here, and everything else is a constant set before upload.
//
// Both amplitudes boot at ZERO. Press w and the robot takes its lean and then stands there;
// nothing oscillates until you dial an amplitude up. That is the point: add one mechanism
// at a time and watch which combination cooperates.
//
//   j / k   leg extension amplitude   -/+ 5 deg     (crank, unipolar 0..amp)
//   n / m   leg swing amplitude       -/+ 2 deg     (hip, half-rectified)
//   , / .   hip_phi                   -/+ 10 deg    (hip phase RELATIVE TO the crank)
//   [ / ]   frequency                 -/+ 0.05 Hz   (bumpless, safe to turn mid-walk)
//   0       both amplitudes to zero   (stops driving without leaving WALK)
//   r w q   READY / WALK / IDLE       i  re-init motors      p  motor health
//
// On hip_phi, from the GRID-4 c1 sweep at mu=0.5 (455k cells): at 1.95/125/28/20 the
// passing window runs 330 -> 40 with its peak at 0, and NOTHING passes between 70 and 250.
// Pooled over the whole grid, hip_phi 0 holds the fastest cell (net 0.350 m/s) while 260
// holds the most cells that walk at all (41%) but none faster than 0.212. If a tuning
// session sits in the dead zone it will never find out why nothing works, which is why this
// is a live knob and not a constant.
//
// The gait itself is unchanged from the sweep's definition:
//   crank_L = 0.5*A_leg*(1 + sin(p))              crank_R = 0.5*A_leg*(1 + sin(p + pi))
//   hip_L   = off + A_hip*max(0, sin(p+pi+PHI))   hip_R   = off + A_hip*max(0, sin(p+PHI))

// =========================== SET THESE BEFORE UPLOADING ===========================
// None of these are on a key. A value that has to be re-picked after every power cycle is
// a value that will be wrong in half the records.
#include <DynamixelShield.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

const float    KAPPA           = 0.0f;   // torso target = kappa * hip-axis roll. 0 = hold level
const float    KP              = 0.5f;   // measured optimum; 1.0 marginal, 2.0 diverges here
const float    KI              = 0.1f;   // contributes under 1 deg on this robot; near inert
const float    KD              = 0.0f;   // [s] on -d(roll)/dt. 0.05 would match the whole KP term
const float    TORSO_CLAMP_DEG = 25.0f;  // mechanical stop measured +27.8 / -31.0 (torso_rom)
const uint16_t TORSO_CURRENT_MA= 3210;   // register maximum = no cap. Under 830 mA to bite
const uint16_t LEG_PGAIN       = 1600;   // POSITION_P_GAIN on hips+cranks; torso left at default
const float    HIP_OFF_DEG     = 20.0f;  // forward lean during walk. 10 sat the robot backwards
const float    T_RAMP = 4.0f, T_SETTLE = 6.0f, T_BLEND = 4.0f;   // staged start [s]
const float    HIP_REST_DEG    = 10.0f;  // rest lean; the ramp starts from here
const uint16_t TEL_MS          = 50;     // telemetry period [ms] -> 20 Hz

const float STEP_LEG = 5.0f, STEP_HIP = 2.0f, STEP_PHI = 10.0f, STEP_FREQ = 0.05f;
const float FREQ_MIN = 0.30f, FREQ_MAX = 2.60f;
// ==================================================================================

// live, on keys. Amplitudes start at zero.
float p_legFreq = 1.50f;      // [Hz]
float p_legAmp  = 0.0f;       // [deg] crank amplitude
float p_hipAmp  = 0.0f;       // [deg] hip swing amplitude
float p_hipPhi  = 0.0f;       // [deg] hip phase relative to the crank

// Changing frequency mid-walk is not free: phase = 2*pi*f*(t - t0), so raising f at
// t - t0 = 13 s by 0.05 jumps the phase by 0.65 of a cycle and kicks the robot. setFreq()
// absorbs the difference into this offset; at a constant frequency it stays put.
float gait_phi_off = 0.0f;

const uint8_t XM_LEFT_SLIDE = 4, XM_RIGHT_SLIDE = 3;
const uint8_t XM_LEFT_HIP   = 2, XM_RIGHT_HIP   = 1, XM_TORSO_ROLL = 0;
const uint8_t MOTOR_IDS[] = {XM_LEFT_SLIDE, XM_RIGHT_SLIDE, XM_LEFT_HIP, XM_RIGHT_HIP, XM_TORSO_ROLL};
const int     MOTOR_COUNT = 5;
const uint8_t LEG_IDS[4]  = {XM_RIGHT_HIP, XM_LEFT_HIP, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};

DynamixelShield  dxl;
Adafruit_BNO055  bno = Adafruit_BNO055(55, 0x28);

enum RobotState { STATE_IDLE, STATE_READY_SLIDE, STATE_READY_HIP, STATE_READY_TORSO, STATE_WALK };
RobotState robot_state = STATE_IDLE;

float         imu_roll = 0, imu_pitch = 0;
float         roll_rate = 0.0f;          // EMA of d(imu_roll)/dt [deg/s], feeds the D term
float         home_deg[MOTOR_COUNT];
unsigned long walk_start_ms = 0;
float         torso_iErr = 0.0f;
float         loop_dt_ema = 0.0f;
unsigned long loop_count = 0;

const float READY_STEP = 1.0f, READY_STEP_TORSO = 1.5f, ARRIVE_THRESH = 1.0f;

const char *CSV_HEADER =
  "w,t,alpha,goal_slL,pos_slL,goal_slR,pos_slR,goal_hipL,pos_hipL,goal_hipR,pos_hipR,"
  "goal_torso,pos_torso,mA_torso,imu_roll,imu_pitch,axis,dt_ms,freq,leg_amp,hip_amp,"
  "hip_phi,hip_off";


// ------------------------------------------------------------------ small helpers
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

void setFreq(float f) {
  f = constrain(f, FREQ_MIN, FREQ_MAX);
  if (robot_state == STATE_WALK) {
    float tw = (millis() - walk_start_ms) / 1000.0f - T_RAMP - T_SETTLE;
    if (tw > 0.0f) gait_phi_off += 2.0f * PI * (p_legFreq - f) * tw;   // bumpless
  }
  p_legFreq = f;
}

void printState() {
  DEBUG_SERIAL.print("# freq "); DEBUG_SERIAL.print(p_legFreq, 2);
  DEBUG_SERIAL.print("  leg ");  DEBUG_SERIAL.print(p_legAmp, 0);
  DEBUG_SERIAL.print("  swing ");DEBUG_SERIAL.print(p_hipAmp, 0);
  DEBUG_SERIAL.print("  phi ");  DEBUG_SERIAL.print(p_hipPhi, 0);
  DEBUG_SERIAL.print("  off ");  DEBUG_SERIAL.print(HIP_OFF_DEG, 0);
  // peak joint rates, against the 420-440 deg/s ceiling measured on the bench 2026-08-28
  DEBUG_SERIAL.print("   crank "); DEBUG_SERIAL.print(PI * p_legFreq * p_legAmp, 0);
  DEBUG_SERIAL.print(" deg/s  hip "); DEBUG_SERIAL.print(2.0f * PI * p_legFreq * p_hipAmp, 0);
  DEBUG_SERIAL.println(" deg/s");
}


// ------------------------------------------------------------------ bring-up
// Split out of setup() because the board is powered over USB and the motors are not:
// opening the serial port resets the board, so if the 12 V comes on afterwards setup() has
// already run against dead motors and every torqueOn was lost. Press 'i' to redo it.
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
    home_deg[i] = zeroExtended(boot_deg);
    dxl.torqueOff(id);
    // EXTENDED position accepts negative / >360 goals; plain OP_POSITION silently rejects
    // them. The torso runs current-based position so its torque ceiling is settable, and
    // mode 5 is still multi-turn, so the goals below are unchanged.
    dxl.setOperatingMode(id, (id == XM_TORSO_ROLL) ? OP_CURRENT_BASED_POSITION
                                                   : OP_EXTENDED_POSITION);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);       // no slew limit
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.torqueOn(id);
    if (id == XM_TORSO_ROLL)
      dxl.writeControlTableItem(GOAL_CURRENT, id, (uint16_t)(TORSO_CURRENT_MA / 2.69f));
    dxl.setGoalPosition(id, boot_deg, UNIT_DEGREE);           // hold where it is; no snap
    int32_t tq = dxl.readControlTableItem(TORQUE_ENABLE, id);
    if (tq != 1) all_ok = false;
    if (verbose) {
      DEBUG_SERIAL.print("ID "); DEBUG_SERIAL.print(id);
      DEBUG_SERIAL.print(" boot = "); DEBUG_SERIAL.print(boot_deg, 2);
      DEBUG_SERIAL.print("  torque="); DEBUG_SERIAL.println(tq);
    }
  }
  // The leg gain used to be written only by a key nobody pressed, while the debug line
  // printed 1600 regardless -- so every record before 2026-08-30 ran at the servos' stored
  // gain. Write it here, then READ IT BACK, so the number printed is the number in force.
  for (int i = 0; i < 4; i++) dxl.writeControlTableItem(POSITION_P_GAIN, LEG_IDS[i], LEG_PGAIN);
  if (verbose) {
    DEBUG_SERIAL.print("# leg POSITION_P_GAIN read back:");
    for (int i = 0; i < 4; i++) {
      DEBUG_SERIAL.print(' ');
      DEBUG_SERIAL.print(dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[i]));
    }
    DEBUG_SERIAL.print("  (asked for "); DEBUG_SERIAL.print(LEG_PGAIN); DEBUG_SERIAL.println(")");
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

  DEBUG_SERIAL.println("pengu_tune ready.  r=READY  w=WALK  q=IDLE  i=re-init  p=health");
  DEBUG_SERIAL.println("  j/k leg ext +-5   n/m leg swing +-2   ,/. hip_phi +-10   [/] freq +-0.05");
  printState();
}


// ------------------------------------------------------------------ IMU
// Only what the control law and the log use: Euler roll (the controlled quantity), pitch
// (lean readout), and the roll rate for the D term. The accelerometer and calibration reads
// that used to sit here fed variables nothing ever read.
void update_imu() {
  imu::Vector<3> e = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  imu_roll = e.y(); imu_pitch = e.z();          // same mapping as pengu.ino

  // roll rate by differencing, lightly filtered. tau 30 ms is far below the gait period, so
  // it costs no phase where it matters, and it keeps the 0.07 deg quantisation steps from
  // showing up as spikes. Differencing rather than the gyro because its SIGN is right by
  // construction; gz is the matching gyro axis but only at slope -0.60 (measured), since
  // Euler rates are not body rates at this pitch.
  static unsigned long r_prev_ms = 0;
  static float roll_prev = 0.0f;
  unsigned long r_now = millis();
  if (r_prev_ms) {
    float rdt = (r_now - r_prev_ms) * 0.001f;
    if (rdt > 0.001f && rdt < 0.2f) {
      float w = rdt / (0.03f + rdt);
      roll_rate = (1.0f - w) * roll_rate + w * ((imu_roll - roll_prev) / rdt);
    }
  }
  r_prev_ms = r_now; roll_prev = imu_roll;
}


// ------------------------------------------------------------------ the gait
void run_walk() {
  float t = (millis() - walk_start_ms) / 1000.0f;

  // staged start: ramp the lean in -> settle -> blend the oscillation in
  float off_deg = HIP_REST_DEG + (HIP_OFF_DEG - HIP_REST_DEG) * constrain(t / T_RAMP, 0.0f, 1.0f);
  float alpha   = constrain((t - T_RAMP - T_SETTLE) / T_BLEND, 0.0f, 1.0f);
  float phase   = (t > T_RAMP + T_SETTLE)
                  ? 2.0f * PI * p_legFreq * (t - T_RAMP - T_SETTLE) + gait_phi_off : 0.0f;

  // legs: unipolar antiphase   hips: half-rectified antiphase, offset by hip_phi
  float magL = alpha * 0.5f * p_legAmp * (1.0f + sinf(phase));
  float magR = alpha * 0.5f * p_legAmp * (1.0f + sinf(phase + PI));
  float phi  = p_hipPhi * PI / 180.0f;
  float hipL_deg = off_deg + alpha * p_hipAmp * max(0.0f, sinf(phase + PI + phi));
  float hipR_deg = off_deg + alpha * p_hipAmp * max(0.0f, sinf(phase + phi));

  // torso: target world roll = KAPPA * hip-axis roll. The axis roll is reconstructed as
  // (torso roll - torso joint), which mocap confirmed in 2026-08-29 against directly
  // measured thigh attitude.
  float J_deg = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - home_deg[idxOf(XM_TORSO_ROLL)];
  float axis  = imu_roll - J_deg;
  float err   = KAPPA * axis - imu_roll;
  static unsigned long ctrl_prev_ms = 0;
  unsigned long ctrl_now_ms = millis();
  float ctrl_dt = ctrl_prev_ms ? (ctrl_now_ms - ctrl_prev_ms) * 0.001f : 0.02f;
  ctrl_prev_ms = ctrl_now_ms;
  if (ctrl_dt < 0.001f || ctrl_dt > 0.5f) ctrl_dt = 0.02f;    // ignore pauses / first call
  loop_dt_ema = (loop_dt_ema <= 0.0f) ? ctrl_dt : (0.9f * loop_dt_ema + 0.1f * ctrl_dt);
  torso_iErr = constrain(torso_iErr + err * ctrl_dt, -20.0f, 20.0f);
  float torso_deg = alpha * constrain(
      (KAPPA - 1.0f) * axis + KP * err + KI * torso_iErr + KD * (-roll_rate),
      -TORSO_CLAMP_DEG, TORSO_CLAMP_DEG);

  // sign conventions inherited from pengu.ino: left negative, right positive
  dxl.setGoalPosition(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)]  - magL,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)] + magR,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_HIP,    home_deg[idxOf(XM_LEFT_HIP)]    - hipL_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_HIP,   home_deg[idxOf(XM_RIGHT_HIP)]   + hipR_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_TORSO_ROLL,  home_deg[idxOf(XM_TORSO_ROLL)]  + torso_deg, UNIT_DEGREE);

  // ---- CSV telemetry. Always on; every row carries the five gait parameters, so a
  // ---- capture spanning a whole tuning session needs no notes to interpret.
  static unsigned long tel = 0, telPrev = 0;
  if (millis() - tel >= TEL_MS) {
    unsigned long nowMs = millis();
    float dtms = telPrev ? (float)(nowMs - telPrev) : 0.0f;
    tel = nowMs; telPrev = nowMs;
    float pSlL = dxl.getPresentPosition(XM_LEFT_SLIDE,  UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_SLIDE)];
    float pSlR = dxl.getPresentPosition(XM_RIGHT_SLIDE, UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_SLIDE)];
    float pHpL = dxl.getPresentPosition(XM_LEFT_HIP,    UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_HIP)];
    float pHpR = dxl.getPresentPosition(XM_RIGHT_HIP,   UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_HIP)];
    float mA   = (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, XM_TORSO_ROLL) * 2.69f;
    DEBUG_SERIAL.print("w,");  DEBUG_SERIAL.print(t, 3);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(alpha, 3);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(-magL, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(pSlL, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(magR, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(pSlR, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(-hipL_deg, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(pHpL, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(hipR_deg, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(pHpR, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(torso_deg, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(J_deg, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(mA, 0);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(imu_roll, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(imu_pitch, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(axis, 2);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(dtms, 1);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(p_legFreq, 3);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(p_legAmp, 1);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(p_hipAmp, 1);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(p_hipPhi, 0);
    DEBUG_SERIAL.print(',');   DEBUG_SERIAL.println(HIP_OFF_DEG, 0);
  }

  // 1 Hz status: what is actually being commanded, and is the loop keeping up
  static unsigned long dbg = 0;
  if (millis() - dbg > 1000) { dbg = millis();
    DEBUG_SERIAL.print("t=");     DEBUG_SERIAL.print(t, 1);
    DEBUG_SERIAL.print(" a=");    DEBUG_SERIAL.print(alpha, 2);
    DEBUG_SERIAL.print(" roll="); DEBUG_SERIAL.print(imu_roll, 1);
    DEBUG_SERIAL.print(" axis="); DEBUG_SERIAL.print(axis, 1);
    DEBUG_SERIAL.print(" torso=");DEBUG_SERIAL.print(torso_deg, 1);
    DEBUG_SERIAL.print(" | leg="); DEBUG_SERIAL.print(p_legAmp, 0);
    DEBUG_SERIAL.print(" swing="); DEBUG_SERIAL.print(p_hipAmp, 0);
    DEBUG_SERIAL.print(" phi=");   DEBUG_SERIAL.print(p_hipPhi, 0);
    DEBUG_SERIAL.print(" f=");     DEBUG_SERIAL.print(p_legFreq, 2);
    DEBUG_SERIAL.print(" | loop=");DEBUG_SERIAL.print(1000.0f * loop_dt_ema, 1);
    DEBUG_SERIAL.print("ms  loops/s="); DEBUG_SERIAL.print(loop_count); loop_count = 0;
    DEBUG_SERIAL.print("  V=");
    DEBUG_SERIAL.print(dxl.readControlTableItem(PRESENT_INPUT_VOLTAGE, XM_LEFT_SLIDE) * 0.1f, 1);
    DEBUG_SERIAL.println("V");
  }
}


// ------------------------------------------------------------------ main loop
void loop() {
  update_imu();

  char cmd = 0;
  if (DEBUG_SERIAL.available()) cmd = (char)DEBUG_SERIAL.read();
  switch (cmd) {
    case 'r': robot_state = STATE_READY_SLIDE; DEBUG_SERIAL.println("-> READY"); break;
    case 'w':
      if (robot_state == STATE_IDLE) {
        walk_start_ms = millis(); torso_iErr = 0; gait_phi_off = 0.0f;
        robot_state = STATE_WALK;
        DEBUG_SERIAL.println("-> WALK (staged start: 4 s lean, 6 s settle, 4 s blend)");
        DEBUG_SERIAL.println(CSV_HEADER);      // one header per bout
        printState();
      } else DEBUG_SERIAL.println("Go READY first.");
      break;
    case 'q': robot_state = STATE_IDLE; DEBUG_SERIAL.println("-> IDLE"); break;

    case 'j': p_legAmp = max(0.0f, p_legAmp - STEP_LEG);  printState(); break;
    case 'k': p_legAmp = min(180.0f, p_legAmp + STEP_LEG); printState(); break;
    case 'n': p_hipAmp = max(0.0f, p_hipAmp - STEP_HIP);  printState(); break;
    case 'm': p_hipAmp = min(45.0f, p_hipAmp + STEP_HIP);  printState(); break;
    case ',': p_hipPhi = fmod(p_hipPhi - STEP_PHI + 360.0f, 360.0f); printState(); break;
    case '.': p_hipPhi = fmod(p_hipPhi + STEP_PHI, 360.0f);          printState(); break;
    case '[': setFreq(p_legFreq - STEP_FREQ); printState(); break;
    case ']': setFreq(p_legFreq + STEP_FREQ); printState(); break;
    case '0': p_legAmp = 0.0f; p_hipAmp = 0.0f;
              DEBUG_SERIAL.println("# amplitudes zeroed (still walking, still logging)");
              printState(); break;

    case 'i':
      robot_state = STATE_IDLE;
      DEBUG_SERIAL.println("# re-initialising motors (use this if the 12 V came on after boot)");
      initMotors(true);
      break;
    case 'p':
      for (int i = 0; i < MOTOR_COUNT; i++) {
        uint8_t id = MOTOR_IDS[i];
        DEBUG_SERIAL.print("ID "); DEBUG_SERIAL.print(id);
        DEBUG_SERIAL.print(dxl.ping(id) ? "  ping OK" : "  PING FAIL");
        DEBUG_SERIAL.print("  pos="); DEBUG_SERIAL.print(dxl.getPresentPosition(id, UNIT_DEGREE), 1);
        DEBUG_SERIAL.print("  torque="); DEBUG_SERIAL.print(dxl.readControlTableItem(TORQUE_ENABLE, id));
        DEBUG_SERIAL.print("  pgain="); DEBUG_SERIAL.print(dxl.readControlTableItem(POSITION_P_GAIN, id));
        DEBUG_SERIAL.print("  hwErr="); DEBUG_SERIAL.println(dxl.readControlTableItem(HARDWARE_ERROR_STATUS, id));
      }
      printState();
      break;
  }

  switch (robot_state) {
    case STATE_IDLE: break;
    case STATE_READY_SLIDE:
      stepMotorToward(XM_LEFT_SLIDE,  0.0f, READY_STEP);
      stepMotorToward(XM_RIGHT_SLIDE, 0.0f, READY_STEP);
      if (arrivedAt(XM_LEFT_SLIDE, 0.0f) && arrivedAt(XM_RIGHT_SLIDE, 0.0f)) {
        robot_state = STATE_READY_HIP; DEBUG_SERIAL.println("Slides at zero -> hips (rest lean)");
      }
      break;
    case STATE_READY_HIP:
      stepMotorToward(XM_LEFT_HIP, 360.0f - HIP_REST_DEG, READY_STEP);   // L: -10 deg (fwd)
      stepMotorToward(XM_RIGHT_HIP, HIP_REST_DEG,         READY_STEP);   // R: +10 deg (fwd)
      if (arrivedAt(XM_LEFT_HIP, 360.0f - HIP_REST_DEG) && arrivedAt(XM_RIGHT_HIP, HIP_REST_DEG)) {
        robot_state = STATE_READY_TORSO; DEBUG_SERIAL.println("Hips at rest lean -> torso");
      }
      break;
    case STATE_READY_TORSO:
      stepMotorToward(XM_TORSO_ROLL, 0.0f, READY_STEP_TORSO);
      if (arrivedAt(XM_TORSO_ROLL, 0.0f)) {
        robot_state = STATE_IDLE; DEBUG_SERIAL.println("READY done -> IDLE");
      }
      break;
    case STATE_WALK: run_walk(); break;
  }

  // No pacing delay: the gait phase comes from absolute time, so a delay would change the
  // control rate without changing the waveform. Measured 57-87 Hz.
  loop_count++;
}
