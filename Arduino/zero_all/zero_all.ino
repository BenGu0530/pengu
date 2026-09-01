// zero_all.ino — walk every motor slowly back to its Dynamixel ABSOLUTE zero and hold
// there, so the mechanical parts can be checked (and re-screwed) against a known datum.
//
// The point of the exercise: if a horn, bracket or crank was bolted on one spline off, the
// joint's electrical zero and the part's intended neutral pose no longer coincide. Driving
// every motor to absolute 0 puts the electrical datum where it belongs; whatever does not
// then look neutral is the part that needs re-seating.
//
// SAFE BY CONSTRUCTION, because a wrong zero may point a joint straight into a hard stop:
//   * CURRENT-BASED POSITION control with a gentle cap -- at a mechanical stop the motor
//     simply cannot push harder, it stalls at the torque you chose rather than driving in.
//   * slow: RATE_DPS deg/s, so there is time to hit 'f' or pull power.
//   * one motor at a time by default, in the order slides -> hips -> torso, which unloads
//     the legs before the torso swings.
//   * a joint that stops following (>STALL_ERR for STALL_HOLD_S) is reported and abandoned,
//     not forced.
//
// "Absolute zero" here means the extended-position coordinate of physical 0 nearest to
// where the motor booted -- the same zeroExtended() convention the walking sketches use,
// so this puts the robot exactly where READY would.
//
// Serial 115200. Commands:
//   z   zero every motor, one at a time (slides, hips, torso)
//   a   zero every motor simultaneously (faster, use once you trust the geometry)
//   0..4  zero just that motor ID
//   f   FREE: torque off on all motors, so parts can be moved by hand
//   h   HOLD: torque on, freeze wherever each motor currently is
//   p   report each motor: raw position, position mod 360, offset from absolute zero, mA
//
// After 'f' and re-screwing, press 'p' to see how far each part now sits from zero, then
// 'z' to bring it back.

#include <DynamixelShield.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

DynamixelShield dxl;
const float DXL_PROTOCOL_VERSION = 2.0;

const uint8_t XM_TORSO_ROLL  = 0;
const uint8_t XM_RIGHT_HIP   = 1;
const uint8_t XM_LEFT_HIP    = 2;
const uint8_t XM_RIGHT_SLIDE = 3;
const uint8_t XM_LEFT_SLIDE  = 4;

// slides first (unload the legs), then hips, then the torso
const uint8_t ORDER[5] = {XM_RIGHT_SLIDE, XM_LEFT_SLIDE, XM_RIGHT_HIP, XM_LEFT_HIP, XM_TORSO_ROLL};
const char *NAMES[5]   = {"R slide", "L slide", "R hip", "L hip", "torso"};

const float CURRENT_UNIT_MA = 2.69f;     // XM430 PRESENT_CURRENT / GOAL_CURRENT unit
const float CUR_CAP_MA      = 600.0f;    // ~1.07 N.m. Raise if a joint will not lift itself.
const float RATE_DPS        = 5.0f;      // slow
const float STALL_ERR       = 5.0f;      // deg of following error that counts as "stopped"
const float STALL_HOLD_S    = 0.8f;
const uint32_t STEP_MS      = 20;

float home_deg[5];                       // absolute zero in extended coordinates, by ID
bool  freed = false;

// extended-coord value of physical 0 nearest to cur (same helper as the walking sketches)
float zeroExtended(float cur) {
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  return cur - ((phys > 180.0f) ? phys - 360.0f : phys);
}

float posOf(uint8_t id)  { return dxl.getPresentPosition(id, UNIT_DEGREE); }
float mAOf(uint8_t id)   { return (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, id) * CURRENT_UNIT_MA; }
float offsetOf(uint8_t id) { return posOf(id) - home_deg[id]; }

const char *nameOf(uint8_t id) {
  for (int i = 0; i < 5; i++) if (ORDER[i] == id) return NAMES[i];
  return "?";
}

void torqueAll(bool on) {
  for (int i = 0; i < 5; i++) {
    if (on) dxl.torqueOn(ORDER[i]); else dxl.torqueOff(ORDER[i]);
  }
  freed = !on;
  DEBUG_SERIAL.println(on ? "# HOLD: torque on, motors frozen where they are"
                          : "# FREE: torque off -- parts can be moved by hand");
}

void report() {
  DEBUG_SERIAL.println("# id  name      pos       mod360    offset_from_zero    mA");
  for (int i = 0; i < 5; i++) {
    uint8_t id = ORDER[i];
    float p = posOf(id);
    float m = fmod(p, 360.0f); if (m < 0) m += 360.0f;
    DEBUG_SERIAL.print("#  "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print("  "); DEBUG_SERIAL.print(nameOf(id));
    DEBUG_SERIAL.print("\t"); DEBUG_SERIAL.print(p, 2);
    DEBUG_SERIAL.print("\t"); DEBUG_SERIAL.print(m, 2);
    DEBUG_SERIAL.print("\t\t"); DEBUG_SERIAL.print(p - home_deg[id], 2);
    DEBUG_SERIAL.print("\t\t"); DEBUG_SERIAL.println(mAOf(id), 0);
  }
}

// Ramp one motor to its absolute zero. Returns false if it stalled short of it.
bool zeroOne(uint8_t id) {
  if (freed) { torqueAll(true); }
  float from = posOf(id);
  float to   = home_deg[id];
  float dist = to - from;
  DEBUG_SERIAL.print("# zeroing "); DEBUG_SERIAL.print(nameOf(id));
  DEBUG_SERIAL.print(" (ID "); DEBUG_SERIAL.print(id);
  DEBUG_SERIAL.print("): "); DEBUG_SERIAL.print(from, 2);
  DEBUG_SERIAL.print(" -> "); DEBUG_SERIAL.print(to, 2);
  DEBUG_SERIAL.print("  ("); DEBUG_SERIAL.print(dist, 2);
  DEBUG_SERIAL.print(" deg at "); DEBUG_SERIAL.print(RATE_DPS, 0);
  DEBUG_SERIAL.println(" deg/s)");

  int steps = (int)(fabsf(dist) / RATE_DPS * (1000.0f / STEP_MS));
  if (steps < 1) steps = 1;
  float stallT = 0;
  for (int k = 1; k <= steps; k++) {
    float goal = from + dist * (float)k / steps;
    dxl.setGoalPosition(id, goal, UNIT_DEGREE);
    delay(STEP_MS);

    float p = posOf(id);
    if (fabsf(goal - p) > STALL_ERR) {
      stallT += STEP_MS / 1000.0f;
      if (stallT >= STALL_HOLD_S) {
        DEBUG_SERIAL.print("#   STOPPED at "); DEBUG_SERIAL.print(p - home_deg[id], 2);
        DEBUG_SERIAL.print(" deg from zero, mA="); DEBUG_SERIAL.print(mAOf(id), 0);
        DEBUG_SERIAL.print(" (cap "); DEBUG_SERIAL.print(CUR_CAP_MA, 0);
        DEBUG_SERIAL.println(") -- mechanical stop or a part fouling. Not forced.");
        dxl.setGoalPosition(id, p, UNIT_DEGREE);      // stop pushing
        return false;
      }
    } else {
      stallT = 0;
    }
    if (DEBUG_SERIAL.available()) {
      char c = (char)DEBUG_SERIAL.read();
      if (c == 'f' || c == 'q') {
        dxl.setGoalPosition(id, posOf(id), UNIT_DEGREE);
        DEBUG_SERIAL.println("#   aborted");
        if (c == 'f') torqueAll(false);
        return false;
      }
    }
  }
  DEBUG_SERIAL.print("#   at zero (offset "); DEBUG_SERIAL.print(offsetOf(id), 2);
  DEBUG_SERIAL.print(" deg, mA="); DEBUG_SERIAL.print(mAOf(id), 0);
  DEBUG_SERIAL.println(") -- holding");
  return true;
}

void zeroAllSimultaneous() {
  if (freed) torqueAll(true);
  float from[5], dist[5];
  float worst = 0;
  for (int i = 0; i < 5; i++) {
    uint8_t id = ORDER[i];
    from[i] = posOf(id);
    dist[i] = home_deg[id] - from[i];
    if (fabsf(dist[i]) > worst) worst = fabsf(dist[i]);
  }
  int steps = (int)(worst / RATE_DPS * (1000.0f / STEP_MS));
  if (steps < 1) steps = 1;
  DEBUG_SERIAL.print("# zeroing all five together, longest travel ");
  DEBUG_SERIAL.print(worst, 1); DEBUG_SERIAL.println(" deg");
  for (int k = 1; k <= steps; k++) {
    for (int i = 0; i < 5; i++)
      dxl.setGoalPosition(ORDER[i], from[i] + dist[i] * (float)k / steps, UNIT_DEGREE);
    delay(STEP_MS);
    if (DEBUG_SERIAL.available()) {
      char c = (char)DEBUG_SERIAL.read();
      if (c == 'f' || c == 'q') {
        for (int i = 0; i < 5; i++) dxl.setGoalPosition(ORDER[i], posOf(ORDER[i]), UNIT_DEGREE);
        DEBUG_SERIAL.println("# aborted");
        if (c == 'f') torqueAll(false);
        return;
      }
    }
  }
  DEBUG_SERIAL.println("# all at zero -- holding");
  report();
}

void setup() {
  DEBUG_SERIAL.begin(115200);
  uint32_t t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 3000);

  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  DEBUG_SERIAL.println("# zero_all -- slow, current-capped return to absolute zero");
  for (int i = 0; i < 5; i++) {
    uint8_t id = ORDER[i];
    if (!dxl.ping(id)) { DEBUG_SERIAL.print("# NO RESPONSE from ID "); DEBUG_SERIAL.println(id); continue; }
    float boot = posOf(id);
    home_deg[id] = zeroExtended(boot);
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_CURRENT_BASED_POSITION);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.writeControlTableItem(GOAL_CURRENT, id, (int)(CUR_CAP_MA / CURRENT_UNIT_MA));
    dxl.torqueOn(id);
    dxl.setGoalPosition(id, boot, UNIT_DEGREE);          // hold where it booted, no snap
    DEBUG_SERIAL.print("# ID "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print(" "); DEBUG_SERIAL.print(nameOf(id));
    DEBUG_SERIAL.print("  boot="); DEBUG_SERIAL.print(boot, 2);
    DEBUG_SERIAL.print("  zero="); DEBUG_SERIAL.print(home_deg[id], 2);
    DEBUG_SERIAL.print("  travel="); DEBUG_SERIAL.print(home_deg[id] - boot, 2);
    DEBUG_SERIAL.println(" deg");
  }
  DEBUG_SERIAL.print("# current cap "); DEBUG_SERIAL.print(CUR_CAP_MA, 0);
  DEBUG_SERIAL.print(" mA (~"); DEBUG_SERIAL.print(CUR_CAP_MA * 0.001f * 1.78f, 2);
  DEBUG_SERIAL.println(" N.m), rate 5 deg/s");
  DEBUG_SERIAL.println("# z=zero one-by-one  a=all together  0-4=one motor  f=free  h=hold  p=report");
}

void loop() {
  if (!DEBUG_SERIAL.available()) { delay(5); return; }
  char c = (char)DEBUG_SERIAL.read();
  if (c == '\n' || c == '\r') return;
  DEBUG_SERIAL.print("# key '"); DEBUG_SERIAL.print(c); DEBUG_SERIAL.println("'");

  if (c == 'z') {
    for (int i = 0; i < 5; i++) { zeroOne(ORDER[i]); delay(300); }
    DEBUG_SERIAL.println("# sequence done");
    report();
  } else if (c == 'a') {
    zeroAllSimultaneous();
  } else if (c >= '0' && c <= '4') {
    zeroOne((uint8_t)(c - '0'));
  } else if (c == 'f') {
    torqueAll(false);
    report();
  } else if (c == 'h') {
    for (int i = 0; i < 5; i++) dxl.setGoalPosition(ORDER[i], posOf(ORDER[i]), UNIT_DEGREE);
    torqueAll(true);
  } else if (c == 'p') {
    report();
  } else {
    DEBUG_SERIAL.println("#   z=zero one-by-one  a=all together  0-4=one motor  f=free  h=hold  p=report");
  }
}
