# DR 协议讨论记录（2026-08-26）—— 待另一 session 拍板

背景：K=5 体检（`k5_lessons_for_grid5.md`）之后，Ben 提出"只做 robust region，
不再加扰动；保留 seed 概念但不动初始姿态"。本 memo 记录讨论中厘清的机制事实、
历史依据、自洽选项与待决定项。**本文不做任何决定**；plots session 不碰协议。

## 1. 机制事实（逐行核实）

**K=1 map 的每一次 trial 本身就带全套扰动。** `grid5/grid5_sweep.py:259-265`
（GRID-4 同构，`physics/grid4_sweep.py:162`）：

```python
for r in range(K):        # K=1 → 只有 r=0，每格一次 trial
    rng = np.random.default_rng((i * len(MUS) + mi) * 100 + r)
    gs.FLOOR_MU = float(mu0) * float(rng.uniform(1 - MU_JIT, 1 + MU_JIT))  # μ ±5%
    gs.POSE_JITTER = {"yaw": ±5°, "pitch": ±3°, "lat": ±1cm}               # 姿态
    rr = gs.run_trial(...)
```

先抽扰动、再跑唯一一次。**CSV 里没有任何 nominal 数**。实例：c3@μ0.3 champion
的 map 值 0.660 来自 r=0 抽到的 μ=0.3084 + yaw −0.46°；同格 nominal
（μ=0.300 整、零偏姿）实测 0.052（本 session 复跑）。K=5 topup 只是补 r=1..4
四抽，机制不变。

**无扰动 ⇒ seed 彻底失效。** trial 路径的唯一随机源就是上面 4 个抽签；仿真本身
确定性（每 trial `mj_resetData` 全复位，无噪声注入、无随机相位、无状态携带）。
全部关掉后不同 seed 逐位相同，因此：
- `pass_rate`/`surv_rate` 退化为 {0,1}，`net_fwd_min ≡ net_fwd_mean`；
- `topup_k.py` 变成 4 倍算力空转；
- 选择链 step 4 "+50000 独立确认 seed" 失去意义，需重设计或删除。

## 2. 历史记录（为什么当年加了扰动）

- **2026-07-30（CLAUDE.md）**：`physics/dr_filter.py` 发现 pass/fail 沿 μ 阶梯
  非单调散点（同一步态 0.6 摔、0.5 又能走，22/40 "摔了又恢复"），单次判定不可靠
  → Ben 定下 K 次带扰动取 pass_rate "压散点噪声"。**注意：dr_filter 本身无 jitter**
  ——散点是失败边界附近物理固有的，不是扰动带来的。
- **2026-08-15**：质量抖动因"糊了 COM 设计轴"被删（`grid4_guide.md:44-46`）——
  与本次对姿态扰动的质疑同构的先例。
- **2026-08-26 planning memo**：jitter 标定发现杀手轴是 pitch，但病根是站立姿态，
  **rest lean 5° 已在源头修掉**（"Jitter does not need to be reduced; the start
  pose needed the firmware's rest lean"）。姿态扰动如今剩余作用 ≈ yaw/lat 航向
  偏移 + t_start 方差。
- **跨机 FP 漂移是真的**：c1@μ0.7 摔倒格换机器重跑 ~20% 翻成存活（planning memo）。
  带扰动的 K 平均是现在吸收这种边界噪声的层；纯确定性后边界格变成随机器而定的
  硬币，仅剩 robust region 的空间平均可部分兜底。

## 3. 自洽选项

- **V1：μ-only seed**（Ben 初步倾向）——seed 只抽 μ×U(0.95,1.05)，姿态严格
  nominal。pass_rate/topup/+50000 链全部保持有意义；契合"姿态扰动是人工的"
  直觉与 2026-07-30 压散点初衷。丢失：对初始条件脆弱性的探测
  （c3 nominal 漏触地那类病只有姿态轴能暴露）。
- **V2：全确定性 K=1**——鲁棒性全交 robust region + step 5 fine scan。最省算力
  （整条 topup 链取消），代价：champion 单抽签膨胀回归、跨机 FP 无平均兜底、
  只能报 region 不能报单格。
- 分解实验（可选，先于决定）：抽样格子跑 nominal / μ-only×5 / pose-only×5 /
  全扰动×5，把 pass 散点方差归因到 μ 轴 vs 姿态轴（后台 nice ~4-8h，
  c4@0.3 有全量 K5 真值可对照）。

## 4. 生效路径（若改）

改协议 = 换测量菜谱，新旧数据不可同表。只有两条路：
- **下轮生效**：GRID-5 按冻结协议跑完（保已烧算力 + 与 GRID-4 可比性）；
  新协议定稿为 `grid5-v2`。
- **本轮重启**：五台机器停掉重发；c6 ~27%（569k 行）及其它机器进度作废，
  每 config ~41h 重来，与 GRID-4 的 protocol 对比断裂。需 Ben 明确下令。

**工程雷（无论怎么决定都该修）**：`check_manifest`（`grid5_sweep.py:184-201`）
只查 protocol/config/K/mujoco/slip，**不查 dr 与 start 块**。谁改了 jitter 常数
接着 resume，新旧协议行会静默混进同一 CSV 且 manifest 照旧标旧参数。
修法：dr/start 加入 gate，或改协议时必须 bump protocol 字符串（强制新 CSV）。

## 5. 与分析层的切分

Ben 的原始痛点——champion 被单次抽签抬高、robust region 把 seed-稳但空间窄的
尖峰（c3@0.3）埋掉——**属于分析层，不依赖协议改动**，已有解法待采纳
（`k5_lessons_for_grid5.md` 建议 1/2）：champion 三数并标（K5 mean / net_fwd_min
/ nbhd）、四象限命名（platform-verified / spike-verified / unverified）、
speed 图加 --k5 模式。plots session 可直接做。

## 6. 待拍板清单（另一 session）

1. V1 / V2 / 维持现状；要不要先跑分解实验；
2. 生效范围：下轮 vs 本轮重启；
3. check_manifest 补 dr/start 检查（低风险，建议无论如何做，但 grid5/ 文件
   live 中——只能在所有 shard 停止后或下轮启动前改）；
4. NET_MIN=0.05 是否太宽（nominal 0.052 的瘸腿步态 technically PASS）；
   是否把 nominal 加为第 6 个确定性 repeat（每格 +1 trial，专抓"球心塌陷"）；
5. 分析层建议 1/2 的采纳（可先行）。
