// pengu.ino
#include <DynamixelShield.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <WiFiNINA.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial

// ===================== Motor IDs =====================
const uint8_t XM_LEFT_SLIDE  = 4;
const uint8_t XM_RIGHT_SLIDE = 3;
const uint8_t XM_LEFT_HIP    = 2;
const uint8_t XM_RIGHT_HIP   = 1;
const uint8_t XM_TORSO_ROLL  = 0;

const uint8_t MOTOR_IDS[] = {XM_LEFT_SLIDE, XM_RIGHT_SLIDE, XM_LEFT_HIP, XM_RIGHT_HIP, XM_TORSO_ROLL};
const int MOTOR_COUNT = 5;

DynamixelShield dxl;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

// ===================== State =====================
enum RobotState { STATE_IDLE, STATE_READY_SLIDE, STATE_READY_HIP, STATE_READY_TORSO, STATE_WALK };
RobotState robot_state = STATE_IDLE;

// ===================== IMU globals =====================
float   imu_yaw = 0, imu_roll = 0, imu_pitch = 0;
float   imu_ax  = 0, imu_ay  = 0, imu_az    = 0;
uint8_t cal_sys = 0, cal_gyro = 0, cal_accel = 0, cal_mag = 0;

// ===================== Params =====================
float p_legAmp  = 1.3f;
float p_legFreq = 1.29f;
float p_hipAmp  = 15.0f * PI / 180.0f;
float p_leanAngle = 21.0f;

bool wifi_active = false;

// ===================== Home positions =====================
float home_deg[MOTOR_COUNT];

int idxOf(uint8_t id) {
  for (int i = 0; i < MOTOR_COUNT; i++)
    if (MOTOR_IDS[i] == id) return i;
  return -1;
}

// ===================== Walk state =====================
unsigned long walk_start_ms = 0;

// ===================== Ready step sizes =====================
const float READY_STEP      = 1.0f;   // deg per loop for slide/hip
const float READY_STEP_TORSO = 1.5f;  // deg per loop for torso
const float ARRIVE_THRESH   = 1.0f;   // deg threshold to consider "arrived"

// ===================== Helpers =====================
float shortestDelta(float current_deg, float target_deg) {
  float delta = target_deg - current_deg;
  while (delta >  180.0f) delta -= 360.0f;
  while (delta < -180.0f) delta += 360.0f;
  return delta;
}

// Move motor one step toward target (physical deg), using shortest path.
// Returns new extended-space goal.
float moveToward(float current_extended, float target_physical, float step) {
  float current_physical = fmod(current_extended, 360.0f);
  if (current_physical < 0) current_physical += 360.0f;
  float delta = shortestDelta(current_physical, target_physical);
  if (fabsf(delta) <= step) return current_extended + delta;
  return current_extended + (delta > 0 ? step : -step);
}

bool arrivedAt(uint8_t id, float target_physical) {
  float current = dxl.getPresentPosition(id, UNIT_DEGREE);
  float physical = fmod(current, 360.0f);
  if (physical < 0) physical += 360.0f;
  return fabsf(shortestDelta(physical, target_physical)) < ARRIVE_THRESH;
}

void stepMotorToward(uint8_t id, float target_physical, float step) {
  float current = dxl.getPresentPosition(id, UNIT_DEGREE);
  float goal = moveToward(current, target_physical, step);
  dxl.setGoalPosition(id, goal, UNIT_DEGREE);
}

// ===================== Setup =====================
void setup() {
  DEBUG_SERIAL.begin(115200);
  while (!DEBUG_SERIAL);

  dxl.begin(1000000);
  dxl.setPortProtocolVersion(2.0);

  DEBUG_SERIAL.println("Checking motors...");
  for (int i = 0; i < MOTOR_COUNT; i++) {
    uint8_t id = MOTOR_IDS[i];
    if (!dxl.ping(id)) {
      DEBUG_SERIAL.print("No response ID: "); DEBUG_SERIAL.println(id);
    } else {
      DEBUG_SERIAL.print("Found ID: "); DEBUG_SERIAL.println(id);
    }
  }

  // Home + position mode
  for (int i = 0; i < MOTOR_COUNT; i++) {
    uint8_t id = MOTOR_IDS[i];
    home_deg[i] = dxl.getPresentPosition(id, UNIT_DEGREE);
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_POSITION);
    dxl.torqueOn(id);
    dxl.setGoalPosition(id, home_deg[i], UNIT_DEGREE);
    DEBUG_SERIAL.print("ID "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print(" home = "); DEBUG_SERIAL.println(home_deg[i], 2);
  }

  // IMU
  DEBUG_SERIAL.println("Initializing BNO055...");
  if (!bno.begin()) {
    DEBUG_SERIAL.println("ERROR: BNO055 not detected.");
    while (1);
  }
  bno.setExtCrystalUse(true);
  DEBUG_SERIAL.println("BNO055 ready.");

  // WiFi
  begin_wifi();

  DEBUG_SERIAL.println("Setup complete. r=Ready, w=Walk");
}








// ===================== Loop =====================
void loop() {
  update_imu();

  char cmd = 0;
  if (wifi_active) cmd = update_wifi();
  if (DEBUG_SERIAL.available()) cmd = (char)DEBUG_SERIAL.read();

  // Commands
  switch (cmd) {
    case 'r':
      if (robot_state != STATE_READY_SLIDE &&
          robot_state != STATE_READY_HIP   &&
          robot_state != STATE_READY_TORSO) {
        robot_state = STATE_READY_SLIDE;
        DEBUG_SERIAL.println("-> READY: zeroing slides...");
      }
      break;
    case 'w':
      if (robot_state == STATE_IDLE) {
        walk_start_ms = millis();
        robot_state = STATE_WALK;
        DEBUG_SERIAL.println("-> WALK");
      } else {
        DEBUG_SERIAL.println("Go READY first.");
      }
      break;
    case 'q':
      robot_state = STATE_IDLE;
      DEBUG_SERIAL.println("-> IDLE");
      break;
  }

  // State execution
  switch (robot_state) {
    case STATE_IDLE:
      break;

    // --- Step 1: zero leg extension (left + right slide simultaneously) ---
    case STATE_READY_SLIDE: {
      stepMotorToward(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)],  READY_STEP);
      stepMotorToward(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)], READY_STEP);
      bool doneL = arrivedAt(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)]);
      bool doneR = arrivedAt(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)]);
      if (doneL && doneR) {
        robot_state = STATE_READY_HIP;
        DEBUG_SERIAL.println("Slides done -> zeroing hips...");
      }
      break;
    }

    // --- Step 2: zero hips (left + right simultaneously) ---
    case STATE_READY_HIP: {
      stepMotorToward(XM_LEFT_HIP,  home_deg[idxOf(XM_LEFT_HIP)],  READY_STEP);
      stepMotorToward(XM_RIGHT_HIP, home_deg[idxOf(XM_RIGHT_HIP)], READY_STEP);
      bool doneL = arrivedAt(XM_LEFT_HIP,  home_deg[idxOf(XM_LEFT_HIP)]);
      bool doneR = arrivedAt(XM_RIGHT_HIP, home_deg[idxOf(XM_RIGHT_HIP)]);
      if (doneL && doneR) {
        robot_state = STATE_READY_TORSO;
        DEBUG_SERIAL.println("Hips done -> zeroing torso...");
      }
      break;
    }

    // --- Step 3: zero torso ---
    case STATE_READY_TORSO: {
      stepMotorToward(XM_TORSO_ROLL, home_deg[idxOf(XM_TORSO_ROLL)], READY_STEP_TORSO);
      if (arrivedAt(XM_TORSO_ROLL, home_deg[idxOf(XM_TORSO_ROLL)])) {
        robot_state = STATE_IDLE;
        DEBUG_SERIAL.println("Torso done -> IDLE (holding).");
      }
      break;
    }

    // --- Walk ---
    case STATE_WALK:
      run_walk();
      break;
  }

  delay(20);
}











// ===================== WALK =====================
void run_walk() {
  float t = (millis() - walk_start_ms) / 1000.0f;

  float s = sinf(2.0f * PI * p_legFreq * t);

  // Leg extension (complementary, around absolute 0)
  float A_leg = p_legAmp * 180.0f / PI;
  float magL  = 0.5f * A_leg * (1.0f + s);
  float magR  = A_leg - magL;
  float LlegGoal = -magL;
  float RlegGoal = +magR;

  // Hip pitch (complementary, same phase)
  float A_hip = p_hipAmp * 180.0f / PI;
  float magHL = 0.5f * A_hip * (1.0f + s);
  float magHR = A_hip - magHL;
  float LhipGoal = +magHL - p_leanAngle;
  float RhipGoal = -magHR + p_leanAngle;

  // Torso
  float torsoGoal = 0.0f;

  dxl.setGoalPosition(XM_LEFT_SLIDE,  LlegGoal,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_SLIDE, RlegGoal,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_HIP,    LhipGoal,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_HIP,   RhipGoal,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_TORSO_ROLL,  torsoGoal, UNIT_DEGREE);
}

// ===================== IMU =====================
void update_imu() {
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  bno.getCalibration(&cal_sys, &cal_gyro, &cal_accel, &cal_mag);
  imu_yaw   = euler.x(); imu_roll  = euler.y(); imu_pitch = euler.z();
  imu_ax    = accel.x(); imu_ay    = accel.y(); imu_az    = accel.z();
}