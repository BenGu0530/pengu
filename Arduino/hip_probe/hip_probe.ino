// hip_probe.ino — the hip version of motor_probe, plus real torque via present current.
//
// The cranks turned out to track their command to ~0.91 while loaded, so crank speed is
// not where the speed gap lives. The hips are the untested pair, and they are the ones
// carrying the body: the sim rollout of the c6 champion puts hip torque at p95 = 2.3-2.7
// N.m with peaks touching the 4.1 N.m stall limit, which is a completely different
// operating regime from the cranks (1.1 N.m, nowhere near stall).
//
// So this sketch adds what motor_probe could not answer: PRESENT_CURRENT is logged
// alongside goal/present position, which turns the log into actual (torque, speed)
// operating points -- directly comparable to the sim's actuator_force, instead of a
// datasheet line.
//
// Waveform: the gait's hip signal is HALF-RECTIFIED, not a plain sine --
//     hip = off + A * max(0, sin(2*pi*f*t + phi))          (L is pi out of phase with R)
// so the peak demanded rate is 2*pi*f*A (at the rectified sine's rising zero crossing),
// twice the rate a plain sine of the same amplitude would need. Copied verbatim from the
// champ sketches so the numbers mean something.
//
// >>> LOADING MATTERS HERE. Run it BOTH ways: <<<
//   1. HANGING (feet off the ground): inertia only -- the best case, like motor_probe.
//   2. STANDING, someone holding the torso so the robot cannot topple or walk away:
//      now the hips push the body's weight, which is the regime the sim says approaches
//      stall. Expect the ratio to drop if torque is the binding constraint.
// Run 'g' and 'a' in each condition and note which was which.
//
// No IMU, no WiFi, no cranks, no torso. Only the two hip motors ever move.
//
// Serial 115200. Commands:
//   g   frequency ladder at AMP_DEG
//   a   amplitude ladder at FREQ_FIXED
//   s   single point (FREQ_FIXED / AMP_DEG)
//   q   stop immediately, return to the rest lean and hold
//   p   motor health: ping / present position / torque enable / hardware error
//   h   reprint the CSV header

#include <DynamixelShield.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

DynamixelShield dxl;
const float DXL_PROTOCOL_VERSION = 2.0;

// Same IDs as pengu.ino / the champ sketches
const uint8_t XM_RIGHT_HIP = 1;
const uint8_t XM_LEFT_HIP  = 2;
const uint8_t IDS[2] = {XM_RIGHT_HIP, XM_LEFT_HIP};   // [0] = R, [1] = L

// Everything that is NOT being measured is held rigid at absolute zero (Ben, 2026-08-28):
// a torque-off torso is a free-swinging dead weight that loads the hips through its own
// inertia, and limp cranks let the legs fold under the standing test. Locked joints are
// commanded ONCE and then left alone -- a position servo holds its last goal -- so the
// lock costs nothing in the sampling loop.
const uint8_t XM_TORSO_ROLL  = 0;
const uint8_t XM_RIGHT_SLIDE = 3;
const uint8_t XM_LEFT_SLIDE  = 4;
const uint8_t LOCK_IDS[3] = {XM_TORSO_ROLL, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};
float lock_home[3] = {0, 0, 0};
bool  locked = false;
const float T_LOCK = 2.0f;        // seconds to ramp the locks from boot pose to zero

// XM430-W350: PRESENT_CURRENT unit is 2.69 mA; stall 4.1 N.m at 2.3 A -> ~1.78 N.m/A.
// The constant is nominal (no no-load-current subtraction), so treat the torque column
// as an estimate and the mA column as the measurement.
const float CURRENT_UNIT_MA = 2.69f;
const float NM_PER_AMP      = 1.78f;

// ---- test points --------------------------------------------------------------------
// Demanded peak rate is 2*pi*f*A (half-rectified). At A=24: 1.0Hz->151  1.67->252  2.0->302
const float FREQS[] = {0.50f, 0.80f, 1.00f, 1.20f, 1.37f, 1.67f, 2.00f};
const int   N_FREQS = 7;
const float AMPS[]  = {12.0f, 16.0f, 20.0f, 24.0f, 28.0f, 32.0f};
const int   N_AMPS  = 6;
const float AMP_DEG    = 24.0f;   // c6 champion hip_amp
const float FREQ_FIXED = 1.67f;   // c6 champion freq
const float HIP_OFF    = 20.0f;   // c6 champion hip_off (the forward lean the swing rides on)
const float T_POINT    = 4.0f;    // seconds per test point
const float T_GAP      = 1.5f;    // seconds held at the offset between points

float home_deg[2] = {0, 0};

// extended-coord value of physical 0 nearest to cur (same helper as the champ sketches)
float zeroExtended(float cur) {
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  return cur - ((phys > 180.0f) ? phys - 360.0f : phys);
}

void initMotor(uint8_t id, float *home_out) {
  if (!dxl.ping(id)) { DEBUG_SERIAL.print("# No response ID "); DEBUG_SERIAL.println(id); }
  float boot = dxl.getPresentPosition(id, UNIT_DEGREE);
  *home_out = zeroExtended(boot);
  dxl.torqueOff(id);
  dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
  dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
  dxl.torqueOn(id);
  dxl.setGoalPosition(id, boot, UNIT_DEGREE);          // hold boot pose, no snap
  DEBUG_SERIAL.print("# ID "); DEBUG_SERIAL.print(id);
  DEBUG_SERIAL.print(" boot="); DEBUG_SERIAL.print(boot, 2);
  DEBUG_SERIAL.print(" home="); DEBUG_SERIAL.println(*home_out, 2);
}

// Ramp torso + both cranks from wherever they booted to absolute zero, then hold there.
// Ramped, not snapped: a torque-on jump to zero from an arbitrary boot pose can be a
// large sudden move.
void lockToZero() {
  float from[3];
  for (int i = 0; i < 3; i++)
    from[i] = dxl.getPresentPosition(LOCK_IDS[i], UNIT_DEGREE);
  int steps = (int)(T_LOCK * 50);
  for (int k = 1; k <= steps; k++) {
    float u = (float)k / steps;
    for (int i = 0; i < 3; i++)
      dxl.setGoalPosition(LOCK_IDS[i], from[i] + u * (lock_home[i] - from[i]), UNIT_DEGREE);
    delay(20);
  }
  locked = true;
  DEBUG_SERIAL.print("# locked: torso/cranks held at absolute zero  (pos =");
  for (int i = 0; i < 3; i++) {
    DEBUG_SERIAL.print(' ');
    DEBUG_SERIAL.print(dxl.getPresentPosition(LOCK_IDS[i], UNIT_DEGREE) - lock_home[i], 2);
  }
  DEBUG_SERIAL.println(" deg from zero)");
}

// R gets +offset, L gets -offset -- the sign convention of the champ sketches
void setHips(float rDeg, float lDeg) {
  dxl.setGoalPosition(XM_RIGHT_HIP, home_deg[0] + rDeg, UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_HIP,  home_deg[1] - lDeg, UNIT_DEGREE);
}

void printHeader() {
  DEBUG_SERIAL.println("# hip_probe v1 -- hip goal vs present + present current");
  DEBUG_SERIAL.println("# waveform: hip = off + A*max(0,sin(2*pi*f*t+phi)), L is pi out of phase with R");
  DEBUG_SERIAL.println("# demanded peak rate = 2*pi*f*A deg/s (half-rectified doubles it vs a plain sine)");
  DEBUG_SERIAL.println("# cur_* in mA; torque_est = A * 1.78 N.m/A (nominal, no no-load subtraction)");
  DEBUG_SERIAL.println("# torso + both cranks are LOCKED at absolute zero for the whole run");
  DEBUG_SERIAL.println("label,f,A,t,goalR,posR,curR_mA,goalL,posL,curL_mA,dt_ms");
}

float readCurrent_mA(uint8_t id) {
  int16_t raw = (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, id);
  return raw * CURRENT_UNIT_MA;
}

void holdOffset(float secs) {
  uint32_t t0 = millis();
  while (millis() - t0 < (uint32_t)(secs * 1000)) {
    setHips(HIP_OFF, HIP_OFF);
    delay(10);
    if (DEBUG_SERIAL.available() && DEBUG_SERIAL.read() == 'q') return;
  }
}

// One test point. Returns false if the user aborted.
bool runPoint(float f, float A) {
  float minR = 1e9, maxR = -1e9, minL = 1e9, maxL = -1e9;
  float prevR = 0, prevL = 0; uint32_t prevMs = 0;
  float peakRateR = 0, peakRateL = 0;
  float peakCurR = 0, peakCurL = 0, sumCurR = 0, sumCurL = 0;
  float dtSum = 0; int nDt = 0, nCur = 0;

  float torso0 = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - lock_home[0];
  float torsoDrift = 0;

  uint32_t t0 = millis();
  bool first = true;
  while (true) {
    uint32_t now = millis();
    float t = (now - t0) / 1000.0f;
    if (t >= T_POINT) break;
    if (DEBUG_SERIAL.available() && DEBUG_SERIAL.read() == 'q') return false;

    float ph = 2.0f * PI * f * t;
    float swR = A * max(0.0f, sinf(ph));
    float swL = A * max(0.0f, sinf(ph + PI));
    float goalR = HIP_OFF + swR, goalL = HIP_OFF + swL;
    setHips(goalR, goalL);

    float posR = dxl.getPresentPosition(XM_RIGHT_HIP, UNIT_DEGREE);
    float posL = dxl.getPresentPosition(XM_LEFT_HIP,  UNIT_DEGREE);
    float curR = readCurrent_mA(XM_RIGHT_HIP);
    float curL = readCurrent_mA(XM_LEFT_HIP);

    uint32_t nowEnd = millis();
    float dt = (nowEnd - prevMs) / 1000.0f;
    if (!first && dt > 1e-4f) {
      if (t > 0.5f / f) {                 // skip the first half cycle (catching up from hold)
        float rR = fabsf(posR - prevR) / dt, rL = fabsf(posL - prevL) / dt;
        if (rR > peakRateR) peakRateR = rR;
        if (rL > peakRateL) peakRateL = rL;
        if (posR < minR) minR = posR;  if (posR > maxR) maxR = posR;
        if (posL < minL) minL = posL;  if (posL > maxL) maxL = posL;
        if (fabsf(curR) > peakCurR) peakCurR = fabsf(curR);
        if (fabsf(curL) > peakCurL) peakCurL = fabsf(curL);
        sumCurR += fabsf(curR); sumCurL += fabsf(curL); nCur++;
      }
      dtSum += dt; nDt++;
    }
    prevR = posR; prevL = posL; prevMs = nowEnd; first = false;

    DEBUG_SERIAL.print("d,");    DEBUG_SERIAL.print(f, 2);     DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(A, 1);    DEBUG_SERIAL.print(',');      DEBUG_SERIAL.print(t, 3);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(goalR, 2); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(posR, 2); DEBUG_SERIAL.print(',');      DEBUG_SERIAL.print(curR, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.print(goalL, 2); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(posL, 2); DEBUG_SERIAL.print(',');      DEBUG_SERIAL.print(curL, 1);
    DEBUG_SERIAL.print(',');     DEBUG_SERIAL.println(dt * 1000.0f, 2);
  }

  torsoDrift = (dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - lock_home[0]) - torso0;
  float ampR = maxR - minR, ampL = maxL - minL;
  DEBUG_SERIAL.print("#SUM f="); DEBUG_SERIAL.print(f, 2);
  DEBUG_SERIAL.print(" A_cmd="); DEBUG_SERIAL.print(A, 1);
  DEBUG_SERIAL.print(" demand_rate="); DEBUG_SERIAL.print(2.0f * PI * f * A, 0);
  DEBUG_SERIAL.print(" | A_R=");  DEBUG_SERIAL.print(ampR, 1);
  DEBUG_SERIAL.print(" A_L=");    DEBUG_SERIAL.print(ampL, 1);
  DEBUG_SERIAL.print(" ratio=");  DEBUG_SERIAL.print(0.5f * (ampR + ampL) / A, 3);
  DEBUG_SERIAL.print(" | peak_rate_R="); DEBUG_SERIAL.print(peakRateR, 0);
  DEBUG_SERIAL.print(" peak_rate_L=");   DEBUG_SERIAL.print(peakRateL, 0);
  DEBUG_SERIAL.print(" | peak_mA_R=");   DEBUG_SERIAL.print(peakCurR, 0);
  DEBUG_SERIAL.print(" peak_mA_L=");     DEBUG_SERIAL.print(peakCurL, 0);
  DEBUG_SERIAL.print(" peak_Nm_R=");     DEBUG_SERIAL.print(peakCurR * 0.001f * NM_PER_AMP, 2);
  DEBUG_SERIAL.print(" peak_Nm_L=");     DEBUG_SERIAL.print(peakCurL * 0.001f * NM_PER_AMP, 2);
  DEBUG_SERIAL.print(" mean_mA=");       DEBUG_SERIAL.print(nCur ? 0.5f * (sumCurR + sumCurL) / nCur : 0.0f, 0);
  DEBUG_SERIAL.print(" | torso_drift="); DEBUG_SERIAL.print(torsoDrift, 2);
  DEBUG_SERIAL.print(" | loop_dt_ms=");  DEBUG_SERIAL.println(nDt ? 1000.0f * dtSum / nDt : 0.0f, 2);
  return true;
}

void setup() {
  DEBUG_SERIAL.begin(115200);
  uint32_t t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 3000);

  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  for (int i = 0; i < 2; i++) {
    DEBUG_SERIAL.print(i == 0 ? "# (R hip) " : "# (L hip) ");
    initMotor(IDS[i], &home_deg[i]);
  }
  for (int i = 0; i < 3; i++) {
    DEBUG_SERIAL.print(i == 0 ? "# (torso) " : (i == 1 ? "# (R crank) " : "# (L crank) "));
    initMotor(LOCK_IDS[i], &lock_home[i]);
  }
  printHeader();
  DEBUG_SERIAL.print("# hip_off="); DEBUG_SERIAL.print(HIP_OFF, 0);
  DEBUG_SERIAL.println(" -- HOLD THE ROBOT. l=lock  g=freq ladder  a=amp ladder  s=single  q=stop  p=health");
  DEBUG_SERIAL.println("# torso + cranks lock to absolute zero on the first ladder (or press l first)");
  DEBUG_SERIAL.println("# run the whole thing twice: hanging, then standing while held");
}

void loop() {
  if (!DEBUG_SERIAL.available()) { delay(5); return; }
  char c = (char)DEBUG_SERIAL.read();

  if (c == 'l') {
    lockToZero();
  } else if (c == 'g') {
    if (!locked) lockToZero();
    DEBUG_SERIAL.print("# freq ladder at A="); DEBUG_SERIAL.println(AMP_DEG, 1);
    holdOffset(T_GAP);
    for (int i = 0; i < N_FREQS; i++) {
      if (!runPoint(FREQS[i], AMP_DEG)) { DEBUG_SERIAL.println("# aborted"); break; }
      holdOffset(T_GAP);
    }
    DEBUG_SERIAL.println("# ladder done");
  } else if (c == 'a') {
    if (!locked) lockToZero();
    DEBUG_SERIAL.print("# amp ladder at f="); DEBUG_SERIAL.println(FREQ_FIXED, 2);
    holdOffset(T_GAP);
    for (int i = 0; i < N_AMPS; i++) {
      if (!runPoint(FREQ_FIXED, AMPS[i])) { DEBUG_SERIAL.println("# aborted"); break; }
      holdOffset(T_GAP);
    }
    DEBUG_SERIAL.println("# ladder done");
  } else if (c == 's') {
    if (!locked) lockToZero();
    runPoint(FREQ_FIXED, AMP_DEG);
    holdOffset(T_GAP);
  } else if (c == 'q') {
    holdOffset(0.2f);
    DEBUG_SERIAL.println("# stopped");
  } else if (c == 'h') {
    printHeader();
  } else if (c == 'p') {
    const uint8_t ALL[5] = {XM_RIGHT_HIP, XM_LEFT_HIP, XM_TORSO_ROLL, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};
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
