// webpage.ino

const char WEBPAGE[] PROGMEM = R"html(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Pengu</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;
     min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:20px;gap:16px}

h1{font-size:1.4rem;color:#fff;margin-top:4px}

/* status bar */
.status-bar{display:flex;align-items:center;gap:10px;background:#1a1d27;
            border-radius:10px;padding:10px 16px;width:100%;max-width:500px}
.dot{width:10px;height:10px;border-radius:50%;background:#f44336}
.dot.on{background:#4caf50;box-shadow:0 0 6px #4caf50}
#lat{margin-left:auto;font-size:.75rem;color:#555}

/* control buttons */
.ctrl{display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%;max-width:500px}
.btn{padding:22px;border:none;border-radius:14px;font-size:1.1rem;font-weight:700;
     cursor:pointer;transition:transform .1s,opacity .1s;-webkit-tap-highlight-color:transparent}
.btn:active{transform:scale(.95);opacity:.8}
.btn-ready{background:#2d6a4f;color:#fff}
.btn-walk {background:#1d6fa4;color:#fff}
.btn-ready.active{background:#4caf50;box-shadow:0 0 12px #4caf5066}
.btn-walk.active {background:#2196f3;box-shadow:0 0 12px #2196f366}

/* IMU card */
.card{background:#1a1d27;border-radius:12px;padding:18px;width:100%;max-width:500px}
.card h2{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:14px}
.row{display:flex;align-items:center;margin-bottom:10px}
.lbl{font-size:.8rem;color:#777;width:55px}
.bw{flex:1;height:5px;background:#2a2d3a;border-radius:3px;margin:0 10px}
.b{height:100%;border-radius:3px;transition:width .12s}
.by{background:#5c9eff}.br{background:#ff7c5c}.bp{background:#5cff9e}
.bax{background:#ffd05c}.bay{background:#c05cff}.baz{background:#5cefff}
.val{font-size:.85rem;font-weight:600;width:62px;text-align:right;font-variant-numeric:tabular-nums}

/* cal */
.cal-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:4px}
.ci{text-align:center}
.cl{font-size:.65rem;color:#555;text-transform:uppercase;margin-bottom:4px}
.cv{font-size:1.2rem;font-weight:700}
.c0{color:#f44336}.c1{color:#ff9800}.c2{color:#ffeb3b}.c3{color:#4caf50}

/* params */
.param-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:4px}
.pi{text-align:center;background:#12141c;border-radius:8px;padding:10px 6px}
.pl{font-size:.65rem;color:#555;text-transform:uppercase;margin-bottom:4px}
.pv{font-size:1rem;font-weight:600;color:#aaa}
</style>
</head>
<body>

<h1>🐧 Pengu Control</h1>

<!-- status -->
<div class="status-bar">
  <div class="dot" id="dot"></div>
  <span id="st">Connecting...</span>
  <span id="lat"></span>
</div>

<!-- control -->
<div class="ctrl">
  <button class="btn btn-ready" id="btn-ready" onclick="send('r')">
    🦶 Ready
  </button>
  <button class="btn btn-walk" id="btn-walk" onclick="send('w')">
    🚶 Walk
  </button>
</div>

<!-- IMU euler + accel -->
<div class="card">
  <h2>Orientation</h2>
  <div class="row"><span class="lbl">Yaw</span><div class="bw"><div class="b by" id="by"></div></div><span class="val" id="vy">—</span></div>
  <div class="row"><span class="lbl">Roll</span><div class="bw"><div class="b br" id="br"></div></div><span class="val" id="vr">—</span></div>
  <div class="row"><span class="lbl">Pitch</span><div class="bw"><div class="b bp" id="bp"></div></div><span class="val" id="vp">—</span></div>
  <h2 style="margin-top:14px">Linear Accel (m/s²)</h2>
  <div class="row"><span class="lbl">aX</span><div class="bw"><div class="b bax" id="bax"></div></div><span class="val" id="vax">—</span></div>
  <div class="row"><span class="lbl">aY</span><div class="bw"><div class="b bay" id="bay"></div></div><span class="val" id="vay">—</span></div>
  <div class="row"><span class="lbl">aZ</span><div class="bw"><div class="b baz" id="baz"></div></div><span class="val" id="vaz">—</span></div>
</div>

<!-- calibration -->
<div class="card">
  <h2>Calibration (0=bad · 3=perfect)</h2>
  <div class="cal-row">
    <div class="ci"><div class="cl">System</div><div class="cv c0" id="cs">0</div></div>
    <div class="ci"><div class="cl">Gyro</div><div class="cv c0" id="cgy">0</div></div>
    <div class="ci"><div class="cl">Accel</div><div class="cv c0" id="ca">0</div></div>
    <div class="ci"><div class="cl">Mag</div><div class="cv c0" id="cm">0</div></div>
  </div>
</div>

<!-- params -->
<div class="card">
  <h2>Gait Params</h2>
  <div class="param-grid">
    <div class="pi"><div class="pl">Leg Amp</div><div class="pv" id="p-la">—</div></div>
    <div class="pi"><div class="pl">Leg Freq</div><div class="pv" id="p-lf">—</div></div>
    <div class="pi"><div class="pl">Hip Amp</div><div class="pv" id="p-ha">—</div></div>
  </div>
</div>

<script>
var state = 'idle';  // idle / ready / walk

function send(cmd) {
  fetch('/cmd?key=' + cmd)
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (cmd === 'r') setState('ready');
      if (cmd === 'w') setState('walk');
    })
    .catch(function(){});
}

function setState(s) {
  state = s;
  document.getElementById('btn-ready').classList.toggle('active', s === 'ready');
  document.getElementById('btn-walk').classList.toggle('active',  s === 'walk');
}

function sb(id, v, mn, mx) {
  var pct = Math.min(100, Math.max(0, (v - mn) / (mx - mn) * 100));
  document.getElementById(id).style.width = pct + '%';
}
function sc(id, v) {
  var e = document.getElementById(id);
  e.textContent = v;
  e.className = 'cv c' + v;
}

async function poll() {
  var t = Date.now();
  try {
    var r = await fetch('/data', {signal: AbortSignal.timeout(1500)});
    var d = await r.json();
    var lat = Date.now() - t;

    document.getElementById('dot').className = 'dot on';
    document.getElementById('st').textContent = 'Connected';
    document.getElementById('lat').textContent = lat + ' ms';

    // euler
    document.getElementById('vy').textContent  = d.yaw.toFixed(1)   + '°';
    document.getElementById('vr').textContent  = d.roll.toFixed(1)  + '°';
    document.getElementById('vp').textContent  = d.pitch.toFixed(1) + '°';
    sb('by', d.yaw,   0,   360);
    sb('br', d.roll,  -90, 90);
    sb('bp', d.pitch, -90, 90);

    // accel
    document.getElementById('vax').textContent = d.ax.toFixed(2);
    document.getElementById('vay').textContent = d.ay.toFixed(2);
    document.getElementById('vaz').textContent = d.az.toFixed(2);
    sb('bax', Math.abs(d.ax), 0, 5);
    sb('bay', Math.abs(d.ay), 0, 5);
    sb('baz', Math.abs(d.az), 0, 5);

    // cal
    sc('cs',  d.cal_sys);
    sc('cgy', d.cal_gyro);
    sc('ca',  d.cal_accel);
    sc('cm',  d.cal_mag);

    // params
    document.getElementById('p-la').textContent = d.legAmp.toFixed(2)  + ' rad';
    document.getElementById('p-lf').textContent = d.legFreq.toFixed(2) + ' Hz';
    document.getElementById('p-ha').textContent = d.hipAmp.toFixed(2)  + ' rad';

  } catch(e) {
    document.getElementById('dot').className = 'dot';
    document.getElementById('st').textContent = 'Disconnected';
    document.getElementById('lat').textContent = '';
  }
}

setInterval(poll, 250);  // 4 Hz polling
poll();
</script>
</body>
</html>
)html";