// motor_probe.ino — what crank rate can this robot's XM430s ACTUALLY reach?
//
// The sim has no velocity limit, so the sweep is free to pick gaits demanding
// pi*f*A_leg = 400-700 deg/s. The XM430-W350 datasheet says 46 rpm = 276 deg/s no-load
// at 12 V. Datasheets are not measurements: this sketch asks the motors directly.
//
// Method: drive the two crank motors with the SAME waveform the gait uses,
//     crank = 0.5 * A * (1 + sin(2*pi*f*t))
// at a ladder of frequencies, while logging goal vs present position every loop.
// A position servo that cannot keep up answers with a SHRUNKEN amplitude and a phase
// lag, both of which fall straight out of the log. Achieved/commanded amplitude vs
// demanded rate IS the motor's own speed envelope, measured on this robot, this
// battery, this load.
//
// >>> HANG THE ROBOT. <<< Feet off the ground: that is the lightest load the cranks
// will ever see, so it is the most generous test possible. If the amplitude already
// shrinks while hanging, it can only be worse while walking. Run it loaded afterwards
// (feet down, held upright) to get the walking-load version of the same curve.
//
// No IMU, no WiFi, no gait, no torso. Only the two crank motors ever move.
//
// Serial 115200. Commands:
//   g   run the frequency ladder (see FREQS) at AMP_DEG
//   a   run the amplitude ladder at FREQ_FIXED
//   s   single point: uses FREQ_FIXED / AMP_DEG
//   q   stop immediately and hold position
//   p   motor health: ping / present position / torque / hardware error
//   h   reprint the CSV header
//
// Output: one CSV row per control step (label,f,A,t,goalL,posL,goalR,posR,dt_ms) plus a
// "#SUM" summary line per test point with achieved amplitude and peak rate.

#include <DynamixelShield.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

DynamixelShield dxl;
const float DXL_PROTOCOL_VERSION = 2.0;

// Same IDs as pengu.ino / the champ sketches
const uint8_t XM_LEFT_SLIDE  = 4;
const uint8_t XM_RIGHT_SLIDE = 3;
const uint8_t IDS[2] = {XM_LEFT_SLIDE, XM_RIGHT_SLIDE};

// ---- test points --------------------------------------------------------------------
// Demanded peak rate is pi*f*A. At A=95: 0.5Hz->149  1.0->299  1.37->409  1.67->498  2.0->597
const float FREQS[] = {0.50f, 0.80f, 1.00f, 1.20f, 1.37f, 1.67f, 2.00f};
const int   N_FREQS = 7;
const float AMPS[]  = {40.0f, 60.0f, 80.0f, 95.0f, 115.0f, 135.0f};
const int   N_AMPS  = 6;
const float AMP_DEG    = 95.0f;   // c6 champion leg_amp
const float FREQ_FIXED = 1.67f;   // c6 champion freq
const float T_POINT    = 4.0f;    // seconds per test point
const float T_GAP      = 1.5f;    // seconds held at 0 between points (let it settle)

float home_deg[2] = {0, 0};

// extended-coord value of physical 0 nearest to cur (same helper as the champ sketches)
float zeroExtended(float cur) {
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  return cur - ((phys > 180.0f) ? phys - 360.0f : phys);
}

void printHeader() {
  DEBUG_SERIAL.println("# motor_probe v1 -- crank goal vs present, robot should be HANGING");
  DEBUG_SERIAL.println("# waveform: crank = 0.5*A*(1+sin(2*pi*f*t)), L commanded negative, R positive");
  DEBUG_SERIAL.println("# demanded peak rate = pi*f*A deg/s;  XM430-W350 @12V datasheet no-load = 276");
  DEBUG_SERIAL.println("label,f,A,t,goalL,posL,goalR,posR,dt_ms");
}

void holdZero(float secs) {
  uint32_t t0 = millis();
  while (millis() - t0 < (uint32_t)(secs * 1000)) {
    dxl.setGoalPosition(XM_LEFT_SLIDE,  home_deg[0], UNIT_DEGREE);
    dxl.setGoalPosition(XM_RIGHT_SLIDE, home_deg[1], UNIT_DEGREE);
    delay(10);
    if (DEBUG_SERIAL.available() && DEBUG_SERIAL.read() == 'q') return;
  }
}

// One test point. Returns false if the user aborted.
bool runPoint(float f, float A) {
  float minL = 1e9, maxL = -1e9, minR = 1e9, maxR = -1e9;
  float prevL = 0, prevR = 0; uint32_t prevMs = 0;
  float peakRateL = 0, peakRateR = 0;
  float dtSum = 0; int nDt = 0;

  uint32_t t0 = millis();
  bool first = true;
  while (true) {
    uint32_t now = millis();
    float t = (now - t0) / 1000.0f;
    if (t >= T_POINT) break;
    if (DEBUG_SERIAL.available() && DEBUG_SERIAL.read() == 'q') return false;

    float mag = 0.5f * A * (1.0f + sinf(2.0f * PI * f * t));
    float goalL = home_deg[0] - mag;
    float goalR = home_deg[1] + mag;
    dxl.setGoalPosition(XM_LEFT_SLIDE,  goalL, UNIT_DEGREE);
    dxl.setGoalPosition(XM_RIGHT_SLIDE, goalR, UNIT_DEGREE);

    float posL = dxl.getPresentPosition(XM_LEFT_SLIDE,  UNIT_DEGREE);
    float posR = dxl.getPresentPosition(XM_RIGHT_SLIDE, UNIT_DEGREE);

    uint32_t nowEnd = millis();
    float dt = (nowEnd - prevMs) / 1000.0f;
    if (!first && dt > 1e-4f) {
      float rL = fabsf(posL - prevL) / dt, rR = fabsf(posR - prevR) / dt;
      // ignore the first half cycle: the servo is still catching up from the hold
      if (t > 0.5f / f) {
        if (rL > peakRateL) peakRateL = rL;
        if (rR > peakRateR) peakRateR = rR;
        if (posL < minL) minL = posL;  if (posL > maxL) maxL = posL;
        if (posR < minR) minR = posR;  if (posR > maxR) maxR = posR;
      }
      dtSum += dt; nDt++;
    }
    prevL = posL; prevR = posR; prevMs = nowEnd; first = false;

    DEBUG_SERIAL.print("d,");   DEBUG_SERIAL.print(f, 2);   DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(A, 1);   DEBUG_SERIAL.print(',');    DEBUG_SERIAL.print(t, 3);
    DEBUG_SERIAL.print(',');    DEBUG_SERIAL.print(goalL, 2); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.print(posL, 2); DEBUG_SERIAL.print(',');   DEBUG_SERIAL.print(goalR, 2);
    DEBUG_SERIAL.print(',');    DEBUG_SERIAL.print(posR, 2); DEBUG_SERIAL.print(',');
    DEBUG_SERIAL.println(dt * 1000.0f, 2);
  }

  float ampL = maxL - minL, ampR = maxR - minR;
  DEBUG_SERIAL.print("#SUM f="); DEBUG_SERIAL.print(f, 2);
  DEBUG_SERIAL.print(" A_cmd=");  DEBUG_SERIAL.print(A, 1);
  DEBUG_SERIAL.print(" demand_rate="); DEBUG_SERIAL.print(PI * f * A, 0);
  DEBUG_SERIAL.print(" | A_L=");  DEBUG_SERIAL.print(ampL, 1);
  DEBUG_SERIAL.print(" A_R=");    DEBUG_SERIAL.print(ampR, 1);
  DEBUG_SERIAL.print(" ratio=");  DEBUG_SERIAL.print(0.5f * (ampL + ampR) / A, 3);
  DEBUG_SERIAL.print(" | peak_rate_L="); DEBUG_SERIAL.print(peakRateL, 0);
  DEBUG_SERIAL.print(" peak_rate_R=");   DEBUG_SERIAL.print(peakRateR, 0);
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
    uint8_t id = IDS[i];
    if (!dxl.ping(id)) { DEBUG_SERIAL.print("# No response ID "); DEBUG_SERIAL.println(id); }
    float boot = dxl.getPresentPosition(id, UNIT_DEGREE);
    home_deg[i] = zeroExtended(boot);
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);        // unlimited, same as the gait sketches
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.torqueOn(id);
    dxl.setGoalPosition(id, boot, UNIT_DEGREE);                // hold boot pose, no snap
    DEBUG_SERIAL.print("# ID "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print(" boot="); DEBUG_SERIAL.print(boot, 2);
    DEBUG_SERIAL.print(" home="); DEBUG_SERIAL.println(home_deg[i], 2);
  }
  printHeader();
  DEBUG_SERIAL.println("# HANG THE ROBOT, then: g=freq ladder  a=amp ladder  s=single  q=stop  p=health");
}

void loop() {
  if (!DEBUG_SERIAL.available()) { delay(5); return; }
  char c = (char)DEBUG_SERIAL.read();

  if (c == 'g') {
    DEBUG_SERIAL.print("# freq ladder at A="); DEBUG_SERIAL.println(AMP_DEG, 1);
    holdZero(T_GAP);
    for (int i = 0; i < N_FREQS; i++) {
      if (!runPoint(FREQS[i], AMP_DEG)) { DEBUG_SERIAL.println("# aborted"); break; }
      holdZero(T_GAP);
    }
    DEBUG_SERIAL.println("# ladder done");
  } else if (c == 'a') {
    DEBUG_SERIAL.print("# amp ladder at f="); DEBUG_SERIAL.println(FREQ_FIXED, 2);
    holdZero(T_GAP);
    for (int i = 0; i < N_AMPS; i++) {
      if (!runPoint(FREQ_FIXED, AMPS[i])) { DEBUG_SERIAL.println("# aborted"); break; }
      holdZero(T_GAP);
    }
    DEBUG_SERIAL.println("# ladder done");
  } else if (c == 's') {
    runPoint(FREQ_FIXED, AMP_DEG);
    holdZero(T_GAP);
  } else if (c == 'q') {
    holdZero(0.2f);
    DEBUG_SERIAL.println("# stopped");
  } else if (c == 'h') {
    printHeader();
  } else if (c == 'p') {
    for (int i = 0; i < 2; i++) {
      uint8_t id = IDS[i];
      bool ok = dxl.ping(id);
      DEBUG_SERIAL.print("# ID "); DEBUG_SERIAL.print(id);
      DEBUG_SERIAL.print(ok ? "  ping OK" : "  PING FAIL");
      DEBUG_SERIAL.print("  pos=");    DEBUG_SERIAL.print(dxl.getPresentPosition(id, UNIT_DEGREE), 1);
      DEBUG_SERIAL.print("  torque="); DEBUG_SERIAL.print(dxl.readControlTableItem(TORQUE_ENABLE, id));
      DEBUG_SERIAL.print("  hwErr=");  DEBUG_SERIAL.println(dxl.readControlTableItem(HARDWARE_ERROR_STATUS, id));
    }
  }
}
