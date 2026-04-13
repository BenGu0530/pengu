// wireless.ino
#include <WiFiNINA.h>
#include <utility/wifi_drv.h>

// RGB LED pins on NINA module
const int RGB_RED   = 25;
const int RGB_GREEN = 26;
const int RGB_BLUE  = 27;

WiFiServer server(80);

// ===================== AP Init =====================
void begin_wifi() {
  WiFiDrv::pinMode(RGB_RED,   OUTPUT);
  WiFiDrv::pinMode(RGB_GREEN, OUTPUT);
  WiFiDrv::pinMode(RGB_BLUE,  OUTPUT);
  set_rgb(20, 0, 0);  // red = not connected yet

  if (WiFi.status() == WL_NO_MODULE) {
    DEBUG_SERIAL.println("No WiFi module.");
    return;
  }

  WiFi.config(IPAddress(192, 168, 4, 1));  // fixed IP for AP
  if (WiFi.beginAP("PENGU") != WL_AP_LISTENING) {
    DEBUG_SERIAL.println("WiFi AP failed.");
    return;
  }

  server.begin();
  wifi_active = true;

  DEBUG_SERIAL.println("WiFi AP started.");
  DEBUG_SERIAL.println("SSID: PENGU");
  DEBUG_SERIAL.println("Connect phone to PENGU, then open: http://192.168.4.1");
}

// ===================== RGB helper =====================
void set_rgb(int r, int g, int b) {
  WiFiDrv::analogWrite(RGB_RED,   r);
  WiFiDrv::analogWrite(RGB_GREEN, g);
  WiFiDrv::analogWrite(RGB_BLUE,  b);
}

void update_rgb() {
  switch (WiFi.status()) {
    case WL_AP_LISTENING:
      set_rgb(0, 20, 0);   // green = AP up, no client
      break;
    case WL_AP_CONNECTED:
      // blue blink = client connected
      if (millis() % 500 > 250) set_rgb(0, 0, 20);
      else                       set_rgb(0, 0, 0);
      break;
    default:
      set_rgb(20, 0, 0);   // red = error
      break;
  }
}

// ===================== HTTP handler =====================
// Returns the command char received, or 0 if none
char update_wifi() {
  update_rgb();

  WiFiClient client = server.available();
  if (!client) return 0;

  String request = "";
  unsigned long t = millis();
  while (client.connected() && millis() - t < 1000) {
    if (client.available()) {
      char c = client.read();
      request += c;
      if (request.endsWith("\r\n\r\n")) break;
    }
  }

  char cmd = 0;

  // --- Route: /cmd?key=X  (control commands) ---
  if (request.indexOf("GET /cmd") >= 0) {
    int idx = request.indexOf("key=");
    if (idx >= 0) {
      cmd = request.charAt(idx + 4);
    }
    // Send minimal JSON ack
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: application/json");
    client.println("Connection: close");
    client.println();
    client.println("{\"ok\":true}");

  // --- Route: /data  (IMU + params broadcast) ---
  } else if (request.indexOf("GET /data") >= 0) {
    send_data(client);

  // --- Route: /  (serve webpage) ---
  } else {
    send_webpage(client);
  }

  client.stop();
  return cmd;
}

// ===================== Data broadcast =====================
void send_data(WiFiClient client) {
  // IMU globals defined in EasyPengu.ino
  extern float imu_yaw, imu_roll, imu_pitch;
  extern float imu_ax, imu_ay, imu_az;
  extern uint8_t cal_sys, cal_gyro, cal_accel, cal_mag;
  // Param globals defined in param_storage.ino
  extern float p_legAmp, p_legFreq, p_hipAmp;

  String json = "{";
  json += "\"yaw\":"      + String(imu_yaw,   2) + ",";
  json += "\"roll\":"     + String(imu_roll,  2) + ",";
  json += "\"pitch\":"    + String(imu_pitch, 2) + ",";
  json += "\"ax\":"       + String(imu_ax,    3) + ",";
  json += "\"ay\":"       + String(imu_ay,    3) + ",";
  json += "\"az\":"       + String(imu_az,    3) + ",";
  json += "\"cal_sys\":"   + String(cal_sys)     + ",";
  json += "\"cal_gyro\":"  + String(cal_gyro)    + ",";
  json += "\"cal_accel\":" + String(cal_accel)   + ",";
  json += "\"cal_mag\":"   + String(cal_mag)     + ",";
  json += "\"legAmp\":"   + String(p_legAmp,  3) + ",";
  json += "\"legFreq\":"  + String(p_legFreq, 3) + ",";
  json += "\"hipAmp\":"   + String(p_hipAmp,  3) + "}";

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: application/json");
  client.println("Connection: close");
  client.println();
  client.println(json);
}

// ===================== Webpage =====================
void send_webpage(WiFiClient client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html");
  client.println("Connection: close");
  client.println();
  // HTML defined in webpage.ino
  extern const char WEBPAGE[];
  const int CHUNK = 128;
  int len = strlen_P(WEBPAGE);
  for (int i = 0; i < len; i += CHUNK) {
    char buf[CHUNK + 1];
    int sz = min(CHUNK, len - i);
    strncpy_P(buf, WEBPAGE + i, sz);
    buf[sz] = '\0';
    client.print(buf);
  }
}