"""
plot_gait_signals.py  –  Visualize the 5 gait control signals

A = left leg (crank-L):    sin(phase)                    — RED solid
B = right leg (crank-R):  -sin(phase)                    — BLUE solid
C = left hip swing:        max(0, -sin(phase))           — RED dashed
D = right hip swing:       max(0, sin(phase))            — BLUE dashed
E = torso roll:            sin(phase)                    — BLACK solid thick

KEY PHASING:
  A and D are IN PHASE  (both peak at phase = π/2)
  B and C are IN PHASE  (both peak at phase = 3π/2)
  E is in phase with D  (torso swings toward stance side)

Usage: python plot_gait_signals.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Parameters ───────────────────────────────────────────────────
freq = 1.62       # Hz
t = np.linspace(0, 3 / freq, 1000)  # 3 full cycles
phase = 2 * np.pi * freq * t

# Amplitudes — all different so signals don't overlap
amp_torso = 1.0    # torso — LARGEST
amp_leg   = 0.6    # leg (crank)
amp_hip   = 0.4    # hip swing

# ─── Signals ──────────────────────────────────────────────────────

# A = left leg (crank-L): full sine
A = amp_leg * np.sin(phase)

# B = right leg (crank-R): = -A
B = -A

# D = right hip swing: only when sin(phase) > 0, else 0
#     IN PHASE with A (both use sin(phase), both peak together)
D = amp_hip * np.maximum(0, np.sin(phase))

# C = left hip swing: only when -sin(phase) > 0, else 0
#     IN PHASE with B
C = amp_hip * np.maximum(0, -np.sin(phase))

# E = torso: full sine, in phase with D
#     When D active (right hip swinging up) -> left is stance -> torso swings LEFT
#     When C active (left hip swinging up) -> right is stance -> torso swings RIGHT
E = amp_torso * np.sin(phase)

# ─── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# ── Top: All 5 signals ──
ax = axes[0]
ax.plot(t, A, color='red',   linewidth=2,   linestyle='-',  label='A: left leg (crank-L)')
ax.plot(t, B, color='blue',  linewidth=2,   linestyle='-',  label='B: right leg (crank-R)')
ax.plot(t, C, color='red',   linewidth=2,   linestyle='--', label='C: left hip swing')
ax.plot(t, D, color='blue',  linewidth=2,   linestyle='--', label='D: right hip swing')
ax.plot(t, E, color='black', linewidth=2.5, linestyle='-',  label='E: torso roll')
ax.set_ylabel('Amplitude (normalized)')
ax.set_title('All gait signals  (A↔D in phase, B↔C in phase)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.axhline(0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.3, 1.3)

# ── Middle: Left side — A (left leg) + C (left hip) + E (torso) ──
ax = axes[1]
ax.plot(t, A, color='red',   linewidth=2,   linestyle='-',  label='A: left leg (crank-L)')
ax.plot(t, C, color='red',   linewidth=2,   linestyle='--', label='C: left hip swing')
ax.plot(t, E, color='black', linewidth=2.5, linestyle='-',  label='E: torso roll')
ax.set_ylabel('Amplitude (normalized)')
ax.set_title('Left side view: left leg + left hip + torso', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.axhline(0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.3, 1.3)

# ── Bottom: Right side — B (right leg) + D (right hip) + E (torso) ──
ax = axes[2]
ax.plot(t, B, color='blue',  linewidth=2,   linestyle='-',  label='B: right leg (crank-R)')
ax.plot(t, D, color='blue',  linewidth=2,   linestyle='--', label='D: right hip swing')
ax.plot(t, E, color='black', linewidth=2.5, linestyle='-',  label='E: torso roll')
ax.set_ylabel('Amplitude (normalized)')
ax.set_xlabel('Time [s]')
ax.set_title('Right side view: right leg + right hip + torso', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.axhline(0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.3, 1.3)

plt.tight_layout()
plt.savefig('gait_signals.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n[Saved] gait_signals.png")