# Fable 战略笔记 — ICRA 方向评估(2026-07-04/05)

> 研究方向层面的评估,非代码。晚点回头看。命题 = "penguin 式躯干 over-stance 步态在湿滑地面更稳(GRF 落摩擦锥内)"。

## 总评
核心命题可测、有 story,但目前形状更像"仿真里确认已知生物力学",不是 robotics contribution。
上 ICRA 主会需把重心押在 **机理定律** 或 **控制贡献**,最好加 sim-to-real。

## 新意 — 诚实切分
- **不新**:人在冰面变企鹅步态 = 2009 J.Neurophysiol 既有观察;"重心压支撑脚上方省摩擦" ≈ ZMP/支撑多边形 textbook。审稿人会说 "statics 101"。
- **可能新(押 headline 在此)**:
  1. **质量分布 × 躯干相位 交互** —— "COM 高度决定躯干 roll 帮不帮忙",μ_req vs COM/leg 定律。护城河 = issue #6。
  2. **over_swing 作对照** —— 证明省摩擦特定于 over-stance,隔离机理。
  3. 讲成**动态**论证(躯干 roll 抵消摆动腿侧向动量),躲开"静力学常识"指控。

## 🔴 最大科学风险 — 先做 KILL TEST
效应符号未验证。命题要 `over_stance μ_req < upright`,但物理上可能反:PD 锁竖直反而 shear 最小,任何 roll 都加侧向 GRF。
→ **立刻用 friction_study 拉三模式 μ_req 曲线,确认符号。** 压不过就转向(如"只有高 COM 才受益")。
**符号没确认前,别烧 #6 Onshape 变体 / 硬件。** 零成本,决定后续所有投入。

## 严谨性缺口(审稿人必打)
- 单速度 0.08 m/s 太薄 → 要 μ_req vs 速度曲线族。
- 单 CMA seed(seed=1)无方差 → N seed / N 初始条件,mean±std。
- 接触模型敏感性(solref/solimp/timestep/priority hack)→ sensitivity check,否则 μ 数字被疑。
- min_mu_to_walk 太粗(10 档生存阈值)→ 以 μ_req_p95 领衔,ladder 佐证。
- 开环正弦是弱项;湿滑真问题是反应式(反馈/滑动恢复)。CPG-RL 本可当桥但现解耦。

## 实际价值 / 泛化
- 价值真(配送/救援双足湿滑运动开放问题);"躯干偏向支撑脚降摩擦需求"可落地。
- 泛化被质疑:5-DOF 曲柄滑块企鹅机器人特殊,非通用 biped,需提前论证外推边界。

## 建议排序
1. 本周 kill test(三模式 μ_req 符号)。
2. headline 从"企鹅更防滑"改成"COM 高度决定躯干 roll 是否防滑 —— μ_req 设计律"。
3. 补严谨性三件套:多速度 + 多 seed + 接触敏感性(workshop→ICRA)。
4. 硬件当 support 别当 gate(冲 ICRA 2027 截稿约 2026-09,~2.5 月)。
5. 中期可选控制贡献:CPG-RL reward 纳入 floor μ/滑动信号,学"低 μ 自动加大 over-stance roll" → 从描述性 study 升到 controller contribution。
