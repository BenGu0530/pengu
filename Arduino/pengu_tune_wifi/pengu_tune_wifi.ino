// pengu_tune_wifi.ino — tune the gait by hand, over WiFi. No terminal.
//
// The USB socket on the old board broke when the robot fell and pulled the cable, so there
// is no serial interface here at all: nothing is read from Serial, every control is a
// button at http://192.168.4.1, and the run is recorded into RAM and downloaded
// afterwards instead of being streamed. Nothing is transmitted while the robot walks:
// a radio write inside the control loop would perturb the very period this data is
// meant to measure. Serial is a boot log only (MIRROR_SERIAL 0 drops it).
//
// Both amplitudes boot at ZERO. Press WALK and the robot takes its lean and then stands
// there; nothing oscillates until an amplitude is dialled up. Add one mechanism at a time
// and watch which combination cooperates.
//
//   leg extension amplitude   +-5 deg    crank, unipolar 0..amp
//   leg swing amplitude       +-2 deg    hip, half-rectified
//   hip_phi                   +-10 deg   hip phase RELATIVE TO the crank
//   frequency                 +-0.05 Hz  bumpless, safe to change mid-walk
//   hip_off                   +-5 deg    forward lean, boots at 50
//
// Two measured constraints this sketch exists to respect:
//
// 1. CRANK VELOCITY CEILING 354 +- 4 deg/s. Twelve measurements on 2026-08-30, air and
//    ground pooled, demands from 424 to 613 deg/s, two frequencies: every one executed at
//    354. Below 380 the servo tracks at 0.99 with a constant 19-25 ms lag; above it the
//    amplitude ratio collapses (0.93 -> 0.81 -> 0.69) and the lag grows (39 -> 63 -> 88 ms).
//    Peak crank rate is pi*f*A_leg, so the executable region is  f * A_leg <= 113.
//    Every gait flashed before that measurement sat above the ceiling, which means the
//    robot was never running the gait the sweep selected -- it ran a clipped, delayed one.
//
// 2. hip_off 50. Applying the ceiling to the GRID-4 c1 sweep at mu=0.5 leaves 2804 of the
//    55430 passing cells (5%), and almost all of them sit at hip_off 50 -- a lean far past
//    anything tried so far. The best executable cell is 1.04 / 240 / 85 / 28 / 50 at
//    net 0.1223 (crank 278, hip 183, both well inside). hip_phi 0, the fastest region in
//    simulation, does not survive the ceiling at all: it needs f 1.7-2.0 with A_leg 125.
//
// The gait is unchanged from the sweep's definition:
//   crank_L = 0.5*A_leg*(1 + sin(p))              crank_R = 0.5*A_leg*(1 + sin(p + pi))
//   hip_L   = off + A_hip*max(0, sin(p+pi+PHI))   hip_R   = off + A_hip*max(0, sin(p+PHI))

#include <DynamixelShield.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial
#define MIRROR_SERIAL 1        // boot log only; nothing is ever READ from Serial

// =========================== SET THESE BEFORE UPLOADING ===========================
const float    KAPPA           = 0.0f;   // torso target = kappa * hip-axis roll. 0 = hold level
const float    KP              = 0.5f;   // measured optimum; 1.0 marginal, 2.0 diverges here
const float    KI              = 0.1f;   // contributes under 1 deg on this robot; near inert
const float    KD              = 0.0f;   // [s] on -d(roll)/dt. 0.05 would match the whole KP term
const float    TORSO_CLAMP_DEG = 25.0f;  // mechanical stop measured +27.8 / -31.0 (torso_rom)
const uint16_t TORSO_CURRENT_MA= 3210;   // register maximum = no cap. Under 830 mA to bite
const uint16_t LEG_PGAIN       = 1600;   // POSITION_P_GAIN on hips+cranks; torso left alone
const uint16_t TORSO_PGAIN     = 0;      // 0 = leave the torso servo at whatever it holds.
                                         // Its current draw carries 21-66% of its energy in
                                         // 2.5-4.5 Hz at every drive frequency tried, which
                                         // is the inner loop buzzing, not the kappa loop --
                                         // lowering this is the untested direction.
const float    T_RAMP = 4.0f, T_SETTLE = 6.0f, T_BLEND = 4.0f;   // staged start [s]
const float    HIP_REST_DEG    = 10.0f;  // rest lean; the hip_off ramp starts from here
const uint16_t TEL_MS          = 20;     // telemetry period -> 50 Hz, Nyquist 25 Hz. At 20 Hz
                                         // the 2.5-4.5 Hz peak could not be told from an
                                         // alias of something faster; this settles it.
const float STEP_LEG = 5.0f, STEP_HIP = 2.0f, STEP_PHI = 10.0f;
const float STEP_FREQ = 0.05f, STEP_OFF = 5.0f;
const float FREQ_MIN = 0.30f, FREQ_MAX = 2.60f;
// ==================================================================================

// live, on buttons. Amplitudes start at zero; the lean starts at the sweep's 50.
float p_legFreq = 1.05f;      // [Hz]  1.04 is the best executable cell; 1.05 is on the step grid
float p_legAmp  = 0.0f;       // [deg] crank amplitude
float p_hipAmp  = 0.0f;       // [deg] hip swing amplitude
float p_hipPhi  = 240.0f;     // [deg] hip phase relative to the crank
float p_hipOff  = 50.0f;      // [deg] forward lean

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
unsigned long loop_count = 0, loop_hz = 0;
bool          motors_ok = false;

const float READY_STEP = 1.0f, READY_STEP_TORSO = 1.5f, ARRIVE_THRESH = 1.0f;


// from wireless.ino
void  begin_wifi();
void  update_wifi();
void  set_status_rgb(int state);
int   state_str(char *b);    // both defined below; wireless.ino calls them from
void  apply_cmd(char cmd);   // inside the /cmd handler


// -------------------------------------------------------------- status text (log only)
// Status text goes to the serial mirror only. It is never sent over the air: the page gets
// the state as the reply to its own button press, and the recording is fetched afterwards.
void tx(const char *s) {
#if MIRROR_SERIAL
  DEBUG_SERIAL.println(s);
#else
  (void)s;
#endif
}

// Fixed-point float formatter. snprintf("%f") is not linked in by default on the SAMD core,
// and String would fragment the heap inside a control loop, so numbers are built by hand.
static char *fmt(char *p, float v, uint8_t dp) {
  if (v < 0) { *p++ = '-'; v = -v; }
  long scale = 1;
  for (uint8_t i = 0; i < dp; i++) scale *= 10;
  long n  = (long)(v * scale + 0.5f);
  p += sprintf(p, "%ld", n / scale);
  if (dp) {
    *p++ = '.';
    for (long d = scale / 10; d > 0; d /= 10) *p++ = '0' + (char)((n / d) % 10);
  }
  return p;
}
static char *fmtc(char *p, float v, uint8_t dp) { p = fmt(p, v, dp); *p++ = ','; return p; }


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

// One place builds the state line: it is printed to the log AND returned as the reply to
// every button press, so the page can show what the robot currently has without a second
// endpoint and without the browser ever seeing the telemetry stream.
int state_str(char *b) {
  char *p = b;
  p += sprintf(p, "# freq ");   p = fmt(p, p_legFreq, 2);
  p += sprintf(p, "  leg ");    p = fmt(p, p_legAmp, 0);
  p += sprintf(p, "  swing ");  p = fmt(p, p_hipAmp, 0);
  p += sprintf(p, "  phi ");    p = fmt(p, p_hipPhi, 0);
  p += sprintf(p, "  off ");    p = fmt(p, p_hipOff, 0);
  // peak joint rates against the measured 354 deg/s ceiling
  p += sprintf(p, "   crank "); p = fmt(p, PI * p_legFreq * p_legAmp, 0);
  p += sprintf(p, "/354  hip "); p = fmt(p, 2.0f * PI * p_legFreq * p_hipAmp, 0);
  p += sprintf(p, "/354%s", (PI * p_legFreq * p_legAmp > 354.0f) ? "  OVER" : "");
  p += sprintf(p, "   buffered ");  p = fmt(p, rec_seconds(), 1);
  p += sprintf(p, "s of %d", (int)(REC_MAX * TEL_MS / 1000));
  p += sprintf(p, "s   loop %luHz", loop_hz);
  *p = 0;
  return (int)(p - b);
}

void announce() {
  // A change made mid-bout has to be timestamped into the recording, or the dumped rows
  // would carry the wrong parameters from that point on.
  if (robot_state == STATE_WALK) rec_event(millis() - walk_start_ms);
  char b[192];
  state_str(b);
  tx(b);
}


// ------------------------------------------------------------------ bring-up
// Split out so a button can redo it: the board is powered over USB/battery and the motors
// are not, so if the 12 V comes on after boot every torqueOn is lost. That produced a whole
// session of "READY prints but nothing moves" on 2026-08-30.
bool initMotors() {
  char b[128];
  bool all_ok = true;
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(2.0);
  for (int i = 0; i < MOTOR_COUNT; i++) {
    uint8_t id = MOTOR_IDS[i];
    if (!dxl.ping(id)) {
      all_ok = false;
      sprintf(b, "No response ID %d   <-- is the 12 V on?", id); tx(b);
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
    if (id == XM_TORSO_ROLL) {
      dxl.writeControlTableItem(GOAL_CURRENT, id, (uint16_t)(TORSO_CURRENT_MA / 2.69f));
      if (TORSO_PGAIN) dxl.writeControlTableItem(POSITION_P_GAIN, id, TORSO_PGAIN);
    }
    dxl.setGoalPosition(id, boot_deg, UNIT_DEGREE);           // hold where it is; no snap
    int32_t tq = dxl.readControlTableItem(TORQUE_ENABLE, id);
    if (tq != 1) all_ok = false;
  }
  // This used to be written only by a key nobody pressed while the log printed 1600
  // regardless, so every record before 2026-08-30 ran at the servos' stored gain. Write it,
  // then READ IT BACK, so the number reported is the number in force.
  for (int i = 0; i < 4; i++) dxl.writeControlTableItem(POSITION_P_GAIN, LEG_IDS[i], LEG_PGAIN);
  sprintf(b, "# leg POSITION_P_GAIN read back: %ld %ld %ld %ld  (asked %u)",
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[0]),
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[1]),
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[2]),
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[3]), LEG_PGAIN);
  tx(b);
  tx(all_ok ? "# all 5 motors powered and holding"
            : "# NOT ALL MOTORS ARE HOLDING -- power the 12 V, then press RE-INIT");
  motors_ok = all_ok;
  return all_ok;
}

void setup() {
#if MIRROR_SERIAL
  DEBUG_SERIAL.begin(115200);
  unsigned long t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 1500);   // never block: there may be no cable
#endif
  begin_wifi();
  initMotors();
  if (!bno.begin()) tx("ERROR: BNO055 not detected.");
  else bno.setExtCrystalUse(true);
  tx("# pengu_tune_wifi ready. Join the PENGU network, open http://192.168.4.1");
  announce();
}



// ------------------------------------------------------------- the on-board recorder
// Nothing is transmitted while walking. Samples go into a RAM ring buffer and the whole
// run is fetched afterwards over HTTP, so the control loop never waits on the radio.
//
// Text is far too fat to buffer -- a CSV row is about 200 bytes, which would be four
// seconds of RAM. These records are 21 bytes. What is stored is only what cannot be
// recomputed: every MEASURED quantity, plus the gait `phase` itself. The four commanded
// leg positions are exact functions of phase and the parameters, so rec_dump() rebuilds
// them from the same expressions run_walk() uses -- keeping one copy of the gait maths
// rather than a second one in the analysis script that would silently drift.
//
// The buffer is a RING: when it fills it overwrites the oldest sample. A bout that outlives
// the buffer therefore keeps its LAST N seconds, which is the part containing the fall.
struct __attribute__((packed)) Rec {
  uint16_t t_ms;        // since WALK; wraps at 65.5 s, longer than any bout so far
  uint16_t phase_q;     // gait phase mod 2pi, mapped onto 0..65535
  int16_t  slL, slR;    // measured crank angles, deg x100
  int16_t  hpL, hpR;    // measured hip angles, deg x100
  int16_t  torso;       // measured torso joint, deg x100
  int16_t  gtorso;      // COMMANDED torso, deg x100 -- feedback, so not recomputable
  int16_t  roll, pitch; // IMU, deg x100
  int8_t   mA20;        // torso current / 20 mA, +-2540 mA
};
const int REC_MAX = 570;              // x21 B = 11970 B. 11.4 s at 50 Hz, 22.8 s at 25 Hz
Rec      rec[REC_MAX];
int      rec_head = 0, rec_count = 0;

// Parameter changes are sparse, so they live in their own small table instead of costing
// bytes in every sample. rec_dump interleaves them so each row carries what was in force.
struct Ev { uint32_t t_ms; float freq, leg, hip, phi, off; };
const int EV_MAX = 40;
Ev  ev[EV_MAX];
int ev_count = 0;

void rec_reset() { rec_head = 0; rec_count = 0; ev_count = 0; }

void rec_event(uint32_t t_ms) {
  if (ev_count >= EV_MAX) return;     // the table is a log of what was tried, not a ring:
  ev[ev_count].t_ms = t_ms;           // losing the FIRST settings would be worse
  ev[ev_count].freq = p_legFreq; ev[ev_count].leg = p_legAmp;
  ev[ev_count].hip  = p_hipAmp;  ev[ev_count].phi = p_hipPhi;
  ev[ev_count].off  = p_hipOff;
  ev_count++;
}

static inline int16_t q100(float v) { return (int16_t)constrain(v * 100.0f, -32000.0f, 32000.0f); }

void rec_push(float t, float phase, float slL, float slR, float hpL, float hpR,
              float torso, float gtorso, float mA) {
  Rec &r = rec[rec_head];
  r.t_ms   = (uint16_t)(t * 1000.0f);
  float ph = fmod(phase, 2.0f * PI); if (ph < 0) ph += 2.0f * PI;
  r.phase_q = (uint16_t)(ph * (65535.0f / (2.0f * PI)));
  r.slL = q100(slL); r.slR = q100(slR); r.hpL = q100(hpL); r.hpR = q100(hpR);
  r.torso = q100(torso); r.gtorso = q100(gtorso);
  r.roll = q100(imu_roll); r.pitch = q100(imu_pitch);
  r.mA20 = (int8_t)constrain(mA / 20.0f, -127.0f, 127.0f);
  rec_head = (rec_head + 1) % REC_MAX;
  if (rec_count < REC_MAX) rec_count++;
}

float rec_seconds() { return rec_count * TEL_MS / 1000.0f; }

// Rebuild the CSV. Slow on purpose: the robot is stopped, so there is no deadline, and a
// readable text file that phase_probe.py already parses is worth more than a compact one.
void rec_dump(WiFiClient client) {
  client.print(F("w,t,alpha,goal_slL,pos_slL,goal_slR,pos_slR,goal_hipL,pos_hipL,"
                 "goal_hipR,pos_hipR,goal_torso,pos_torso,mA_torso,imu_roll,imu_pitch,"
                 "axis,dt_ms,freq,leg_amp,hip_amp,hip_phi,hip_off\n"));
  int start = (rec_count == REC_MAX) ? rec_head : 0;     // oldest surviving sample
  int ei = 0;
  float f = ev_count ? ev[0].freq : p_legFreq, A = ev_count ? ev[0].leg : p_legAmp;
  float H = ev_count ? ev[0].hip : p_hipAmp,  P = ev_count ? ev[0].phi : p_hipPhi;
  float O = ev_count ? ev[0].off : p_hipOff;
  uint16_t prev_t = 0;
  char b[256];
  for (int i = 0; i < rec_count; i++) {
    Rec &r = rec[(start + i) % REC_MAX];
    while (ei < ev_count && ev[ei].t_ms <= r.t_ms) {     // parameters in force at this row
      f = ev[ei].freq; A = ev[ei].leg; H = ev[ei].hip; P = ev[ei].phi; O = ev[ei].off; ei++;
    }
    float t     = r.t_ms / 1000.0f;
    float phase = r.phase_q * (2.0f * PI / 65535.0f);
    float alpha = constrain((t - T_RAMP - T_SETTLE) / T_BLEND, 0.0f, 1.0f);
    float off   = HIP_REST_DEG + (O - HIP_REST_DEG) * constrain(t / T_RAMP, 0.0f, 1.0f);
    float phi   = P * PI / 180.0f;
    // the same expressions run_walk() uses, so the commanded columns cannot drift from it
    float magL = alpha * 0.5f * A * (1.0f + sinf(phase));
    float magR = alpha * 0.5f * A * (1.0f + sinf(phase + PI));
    float hL   = off + alpha * H * max(0.0f, sinf(phase + PI + phi));
    float hR   = off + alpha * H * max(0.0f, sinf(phase + phi));
    float roll = r.roll / 100.0f, J = r.torso / 100.0f;
    char *p = b;
    p += sprintf(p, "w,");
    p = fmtc(p, t, 3);              p = fmtc(p, alpha, 3);
    p = fmtc(p, -magL, 2);          p = fmtc(p, r.slL / 100.0f, 2);
    p = fmtc(p, magR, 2);           p = fmtc(p, r.slR / 100.0f, 2);
    p = fmtc(p, -hL, 2);            p = fmtc(p, r.hpL / 100.0f, 2);
    p = fmtc(p, hR, 2);             p = fmtc(p, r.hpR / 100.0f, 2);
    p = fmtc(p, r.gtorso / 100.0f, 2); p = fmtc(p, J, 2);
    p = fmtc(p, r.mA20 * 20.0f, 0); p = fmtc(p, roll, 2);
    p = fmtc(p, r.pitch / 100.0f, 2);  p = fmtc(p, roll - J, 2);
    p = fmtc(p, (float)(uint16_t)(r.t_ms - prev_t), 1);
    p = fmtc(p, f, 3); p = fmtc(p, A, 1); p = fmtc(p, H, 1);
    p = fmtc(p, P, 0); p = fmt(p, O, 0);
    *p++ = '\n'; *p = 0;
    prev_t = r.t_ms;
    client.print(b);
  }
}

// ------------------------------------------------------------------ IMU
// Only what the control law and the log use: Euler roll (the controlled quantity), pitch
// (lean readout), and the roll rate for the D term.
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
  float off_deg = HIP_REST_DEG + (p_hipOff - HIP_REST_DEG) * constrain(t / T_RAMP, 0.0f, 1.0f);
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
  // (torso roll - torso joint), which the 2026-08-29 mocap confirmed against directly
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

  // ---- record into RAM. No radio traffic here: that is the whole point of buffering.
  static unsigned long tel = 0;
  if (millis() - tel >= TEL_MS) {
    tel = millis();
    rec_push(t, phase,
             dxl.getPresentPosition(XM_LEFT_SLIDE,  UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_SLIDE)],
             dxl.getPresentPosition(XM_RIGHT_SLIDE, UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_SLIDE)],
             dxl.getPresentPosition(XM_LEFT_HIP,    UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_HIP)],
             dxl.getPresentPosition(XM_RIGHT_HIP,   UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_HIP)],
             J_deg, torso_deg,
             (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, XM_TORSO_ROLL) * 2.69f);
  }
}


// Commands are applied HERE, not in loop(), so the HTTP reply carries the state the press
// produced rather than the one before it. wireless.ino calls it from the /cmd handler.
void apply_cmd(char cmd) {
  switch (cmd) {
    case 'r': robot_state = STATE_READY_SLIDE; tx("-> READY"); break;
    case 'w':
      if (robot_state == STATE_IDLE) {
        walk_start_ms = millis(); torso_iErr = 0; gait_phi_off = 0.0f;
        rec_reset(); rec_event(0);          // event 0 = the settings the bout started on
        robot_state = STATE_WALK;
        tx("-> WALK (staged start: 4 s lean, 6 s settle, 4 s blend)");
        announce();
      } else tx("Go READY first.");
      break;
    case 'q': robot_state = STATE_IDLE; tx("-> IDLE"); break;

    case 'j': p_legAmp = max(0.0f, p_legAmp - STEP_LEG);   announce(); break;
    case 'k': p_legAmp = min(180.0f, p_legAmp + STEP_LEG); announce(); break;
    case 'n': p_hipAmp = max(0.0f, p_hipAmp - STEP_HIP);   announce(); break;
    case 'm': p_hipAmp = min(45.0f, p_hipAmp + STEP_HIP);  announce(); break;
    case ',': p_hipPhi = fmod(p_hipPhi - STEP_PHI + 360.0f, 360.0f); announce(); break;
    case '.': p_hipPhi = fmod(p_hipPhi + STEP_PHI, 360.0f);          announce(); break;
    case '[': setFreq(p_legFreq - STEP_FREQ); announce(); break;
    case ']': setFreq(p_legFreq + STEP_FREQ); announce(); break;
    case 'o': p_hipOff = max(0.0f,  p_hipOff - STEP_OFF); announce(); break;
    case 'O': p_hipOff = min(70.0f, p_hipOff + STEP_OFF); announce(); break;
    case '0': p_legAmp = 0.0f; p_hipAmp = 0.0f;
              tx("# amplitudes zeroed (still walking, still recording)"); announce(); break;
    case 'i': robot_state = STATE_IDLE; tx("# re-initialising motors"); initMotors(); break;
  }
}


// ------------------------------------------------------------------ main loop
void loop() {
  update_imu();

  update_wifi();

  switch (robot_state) {
    case STATE_IDLE: break;
    case STATE_READY_SLIDE:
      stepMotorToward(XM_LEFT_SLIDE,  0.0f, READY_STEP);
      stepMotorToward(XM_RIGHT_SLIDE, 0.0f, READY_STEP);
      if (arrivedAt(XM_LEFT_SLIDE, 0.0f) && arrivedAt(XM_RIGHT_SLIDE, 0.0f)) {
        robot_state = STATE_READY_HIP; tx("Slides at zero -> hips (rest lean)");
      }
      break;
    case STATE_READY_HIP:
      stepMotorToward(XM_LEFT_HIP, 360.0f - HIP_REST_DEG, READY_STEP);   // L: -10 deg (fwd)
      stepMotorToward(XM_RIGHT_HIP, HIP_REST_DEG,         READY_STEP);   // R: +10 deg (fwd)
      if (arrivedAt(XM_LEFT_HIP, 360.0f - HIP_REST_DEG) && arrivedAt(XM_RIGHT_HIP, HIP_REST_DEG)) {
        robot_state = STATE_READY_TORSO; tx("Hips at rest lean -> torso");
      }
      break;
    case STATE_READY_TORSO:
      stepMotorToward(XM_TORSO_ROLL, 0.0f, READY_STEP_TORSO);
      if (arrivedAt(XM_TORSO_ROLL, 0.0f)) { robot_state = STATE_IDLE; tx("READY done -> IDLE"); }
      break;
    case STATE_WALK: run_walk(); break;
  }

  set_status_rgb(motors_ok ? (robot_state == STATE_WALK ? 2 : 1) : 0);

  // No pacing delay: the gait phase comes from absolute time, so a delay would change the
  // control rate without changing the waveform.
  loop_count++;
  static unsigned long hz_t = 0;
  if (millis() - hz_t >= 1000) { hz_t = millis(); loop_hz = loop_count; loop_count = 0; }
}
