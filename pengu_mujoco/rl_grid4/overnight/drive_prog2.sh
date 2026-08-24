#!/usr/bin/env bash
# Frozen driver. Runs T1, then builds T2 from T1's own numbers, then T3.
# Never edit while running -- bash reads a script by byte offset.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../.sweep_venv/bin/python}
L=overnight/queue_prog2.log

run() { echo "[drive] $1 start $(date '+%F %T')" >> $L
        bash overnight/run_gen_v5.sh "$1" "overnight/cfgs/$1.txt" > "overnight/$1.log" 2>&1
        echo "[drive] $1 done $(date '+%F %T')" >> $L; }

run t1

# --- T2: carry forward whichever straight weight scored better, and pair the
# penalty shape with it. Selection is (ePass, eNfwd, -|stride_asym|) from the
# CONFIRMATION csv, not the training diag.
BEST=$("$PY" - <<'PYEOF'
import csv, os
def score(n):
    f=f"overnight/t1/{n}.eval.csv"
    if not os.path.exists(f): return (-1,-1,-1)
    r=list(csv.DictReader(open(f)))
    if not r: return (-1,-1,-1)
    g=lambda k:[float(x[k]) for x in r if x.get(k) not in (None,"")]
    p=sum(float(x.get("pass",0)) for x in r)/len(r)
    nf=sum(g("net_fwd"))/max(1,len(g("net_fwd")))
    a=g("stride_asym") or [1.0]
    return (p, nf, -abs(sum(a)/len(a)))
best = max([("0.3","str03"),("1.0","str10")], key=lambda kv: score(kv[1]))
print(best[0])
PYEOF
)
echo "[drive] T1 straight winner: $BEST" >> $L
cat > overnight/cfgs/t2.txt <<EOF
# T2 -- built from T1's confirmation numbers by drive_prog2.sh.
# straight=$BEST carried forward as the better of 0.3 / 1.0.
pen_str  --crank-band 0.0 1.9 --shape penalty --rw fall=250 straight=$BEST
pen_s1   --crank-band 0.0 1.9 --shape penalty --rw fall=250 --seed 1
str_s1   --crank-band 0.0 1.9 --rw straight=$BEST --seed 1
EOF
run t2

# --- T3: seed spread on whichever of the two interventions held up, so the
# result is not a single draw (open concern C1).
cat > overnight/cfgs/t3.txt <<EOF
# T3 -- seed spread. C1: a single run's number is a draw, not a measurement.
pen_str_s1 --crank-band 0.0 1.9 --shape penalty --rw fall=250 straight=$BEST --seed 1
pen_str_s2 --crank-band 0.0 1.9 --shape penalty --rw fall=250 straight=$BEST --seed 2
str_s2     --crank-band 0.0 1.9 --rw straight=$BEST --seed 2
EOF
run t3
echo "[drive] all done $(date '+%F %T')" >> $L
