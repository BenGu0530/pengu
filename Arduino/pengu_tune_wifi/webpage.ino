// webpage.ino — the whole interface. There is no terminal any more, so every control the
// tuning session needs is a button here and nothing else is.
//
// What is deliberately NOT on this page: kappa, kp, ki, kd, the torso clamp, the torso
// current limit, the leg P gain, the telemetry rate. Those are decided once and flashed
// (see the block at the top of pengu_tune_wifi.ino). A value that has to be re-picked after
// every power cycle ends up wrong in half the records.
//
// The five pairs that ARE here are the ones a hand-tuning session moves: the two
// amplitudes, the phase between them, the frequency, and the lean. DOWNLOAD is a plain
// link to /dump, so the browser saves the CSV itself and no capture script is needed.

// The Arduino build concatenates the tabs in order: the main sketch, then the others
// alphabetically -- so this file is compiled BEFORE wireless.ino and has to bring in the
// WiFi types itself. Include guards make the second include free.
#include <WiFiNINA.h>

const char WEBPAGE[] PROGMEM = R"HTML(<!doctype html><html><head>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>pengu tune</title><style>
*{box-sizing:border-box}
body{margin:0;padding:12px;background:#111;color:#eee;font:15px/1.35 -apple-system,system-ui,sans-serif}
h1{font-size:1rem;margin:0 0 10px;letter-spacing:.06em;text-transform:uppercase;opacity:.7}
.row{display:flex;gap:8px;margin-bottom:8px}
button{flex:1;padding:16px 8px;border:0;border-radius:10px;background:#2a2f36;color:#eee;
       font-size:1rem;font-weight:600;cursor:pointer;-webkit-tap-highlight-color:transparent}
button:active{background:#3d4550}
a.dl{flex:1;padding:16px 8px;border-radius:10px;background:#26405e;color:#dbeaff;
     font-size:1rem;font-weight:600;text-align:center;text-decoration:none}
.go{background:#1f6f3f}.stop{background:#7a2323}.big{padding:22px 8px;font-size:1.15rem}
.lbl{display:flex;justify-content:space-between;align-items:baseline;
     margin:14px 0 4px;font-size:.75rem;letter-spacing:.09em;text-transform:uppercase;opacity:.55}
.val{font-size:1.05rem;font-weight:700;opacity:1;color:#8fd0ff;text-transform:none;letter-spacing:0}
#st{margin-top:14px;padding:10px;border-radius:8px;background:#1a1e23;
    font:12px/1.5 ui-monospace,Menlo,monospace;color:#9fb3c8;white-space:pre-wrap;min-height:3em}
</style></head><body>
<h1>pengu tune</h1>

<div class="row">
  <button class="go big" onclick="c('r')">READY</button>
  <button class="go big" onclick="c('w')">WALK</button>
</div>
<div class="row">
  <button class="stop" onclick="c('q')">STOP</button>
  <button class="stop" onclick="c('0')">AMPS&nbsp;0</button>
  <button onclick="c('i')">re-init</button>
</div>
<div class="row">
  <a class="dl" href="/dump" download="pengu.csv">DOWNLOAD RECORDING</a>
</div>

<div class="lbl"><span>leg extension</span><span class="val" id="v-leg">&mdash;</span></div>
<div class="row"><button onclick="c('j')">&minus; 5&deg;</button><button onclick="c('k')">+ 5&deg;</button></div>

<div class="lbl"><span>leg swing</span><span class="val" id="v-sw">&mdash;</span></div>
<div class="row"><button onclick="c('n')">&minus; 2&deg;</button><button onclick="c('m')">+ 2&deg;</button></div>

<div class="lbl"><span>hip_phi</span><span class="val" id="v-phi">&mdash;</span></div>
<div class="row"><button onclick="c(',')">&minus; 10&deg;</button><button onclick="c('.')">+ 10&deg;</button></div>

<div class="lbl"><span>frequency</span><span class="val" id="v-f">&mdash;</span></div>
<div class="row"><button onclick="c('[')">&minus; 0.05</button><button onclick="c(']')">+ 0.05</button></div>

<div class="lbl"><span>hip_off (lean)</span><span class="val" id="v-off">&mdash;</span></div>
<div class="row"><button onclick="c('o')">&minus; 5&deg;</button><button onclick="c('O')">+ 5&deg;</button></div>

<div id="st">READY, dial the amplitudes up, WALK. The run is held in the robot's RAM;
press DOWNLOAD after stopping. The buffer is a ring, so an over-long bout keeps its last
seconds -- the part with the fall in it.</div>

<script>
// One request at a time. Every button press is a full HTTP round trip to a board that is
// also running a control loop, so queueing them would pile up latency; a press while one is
// in flight is dropped instead.
var busy = false;
function show(s){
  document.getElementById('st').textContent = s;
  var g = function(k){ var m = s.match(new RegExp(k + "\\s+(-?[0-9.]+)")); return m ? m[1] : null; };
  var f = {'v-leg':'leg','v-sw':'swing','v-phi':'phi','v-f':'freq','v-off':'off'};
  for (var id in f){ var v = g(f[id]); if (v !== null) document.getElementById(id).textContent = v; }
  // the robot appends OVER when pi*f*A_leg exceeds the measured 354 deg/s crank ceiling
  document.getElementById('v-leg').style.color = s.indexOf('OVER') >= 0 ? '#ff8a6b' : '#8fd0ff';
}
function c(k){
  if (busy) return;
  busy = true;
  fetch('/cmd?key=' + encodeURIComponent(k))
    .then(function(r){ return r.text(); })
    .then(function(t){ busy = false; show(t); })
    .catch(function(){ busy = false; document.getElementById('st').textContent = 'no reply'; });
}
window.addEventListener('load', function(){ c('?'); });   // unknown key: just fetches state
</script></body></html>)HTML";

void send_webpage(WiFiClient client) {
  client.print(F("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"));
  const int CHUNK = 256;
  int len = strlen_P(WEBPAGE);
  char buf[CHUNK + 1];
  for (int i = 0; i < len; i += CHUNK) {
    int sz = min(CHUNK, len - i);
    memcpy_P(buf, WEBPAGE + i, sz);
    buf[sz] = 0;
    client.print(buf);
  }
}
