# Plot pipeline session 记录（2026-08-26）

本 session 定位：**plots only**——搭 GRID-5 的画图管线、冻结视觉风格、用旧数据
（GRID-4 / fig45）把每张图的样式跑通。风格阶段不看趋势、不下结论；正式趋势图
等 grid5-v2 数据到齐后用同一批脚本直接出。

## 1. 冻结的视觉契约（Ben 拍板）

`docs/grid5/PLOT_STYLE.md` + `grid5/analysis/style5.py`（唯一真源）：
- **颜色 + 线型 = gait**：κ=0 蓝实线 + 实心 marker；κ=2 红虚线 + 空心 marker
  （黑白打印不丢信息；κ=1 绿点划、无 torso 灰点线预留）
- **marker 形状 = COM**：○1.05 △1.10 □1.20 ◇1.31 ▽1.40（P 1.60 预留）
- 无 shade 渐变（Ben 否决）；marker 大（ms=10）保重叠可读
- 每图必经 `style5.finish()`：footer 戳 K/tier/mean-vs-best + 自动灰度孪生进 bw/
- legend：`legend_combined`（gait+COM 一框，默认右上）
- 风格阶段不标 n=（数据完整性标注等正式图再定）

## 2. 基础设施

- `load5.py`：manifest 驱动 loader（**grid5-v1/v2 双协议**，v2.3 网格
  4,147,200 行/config 自动适应）、npz cache（静态文件二次加载 <1s）、
  partial 数据 `present` 追踪、fall_phase 拆 4 个 int8 平面、
  seam 正确的 nbhd（valid-contributor 除法）、`load_topup()` 读权威 K=5
- `validate_grid4.py`：新 loader 对 INDEX.md 已发布 robust volume
  6 config × 4 μ 逐格复现 PASS（legacy 与 seam-aware 双口径）
- `mech_common.py`：机制图共享底座（建仿真、受力、支撑中心、针孔投影）

## 3. 图库（grid5/analysis/figs/，一图一脚本）

**map 读取类**（`--round grid4|grid5`）：
- `robust_region.py`（F2）、`com_ladder.py`（F6）、`fall_phase.py`（F8，v2 partial 首验）
- `speed_vs_mu.py`（F4：`--tier pass|robust`、`--top N`、`--single`、`--ylim`）
- `speed_rank.py`（尖峰 vs 平台诊断，c3@0.3 断崖的出处）
- `cot_vs_mu.py`（标题 Mechanical cost of transport；grid5 用 map cot_net，
  grid4 fallback 到 fig45 重跑的 cot_pos——两口径不许同图）
- `duty_factor.py`（时间口径：D = double_t + single_t/2，
  single = 1−|2D−1|、double = max(2D−1,0)、aerial = max(1−2D,0)；
  柱状 4μ 面板 + violin `--mu`；资格 = robust finalists 幸存者）

**机制插图类**（跑一段短 nominal 仿真，默认 gait = c6@0.1 verified 直行格
f1.67/φ340/leg95/hip24/off20，5/5、head 0.993）：
- `com_tick.py`（正面 pendulum 视图 + 顶视三姿态叠影，`--bg floor|white|light`，
  相位百分比标注在左缘）
- `duty_strip.py`（9 帧顶视走带 + duty 条 + COM 侧移曲线）
- `torso_lat.py` / `attitude_phase.py`（侧摆时序；roll/pitch/yaw 相位图，
  重力法相对 rest pose——torso body 局部系非 z-up，旧的 tz·left 法量错轴已弃）
- `foot_clearance.py`（`--style stride`（人类步态式：垂直 vs 水平位移）/
  `--style phase`；基线=受载均值，负值是滚动足几何非穿透——实测最深接触穿透
  −5.9 mm 瞬态；基线改"每步起点归零"留待后续）

## 4. 数据侦查与更正（细节见 k5_lessons_for_grid5.md）

- c3@μ0.3 的 0.660：双格尖峰、但 K=5 verified（5/5, 0.490, min 0.251）；
  c4@0.3 全量 topup 里有更硬的 0.535/min 0.500 平台带——**"topup 只做 top-20
  打假不能寻真"**（真 champ 藏在 K1 榜 1.5万–4.3万名）
- **手搓 replay ≠ topup_k.py**（同 seed 公式结果不同）：K=5 数字只认 topup 产物
- champion K=5 pass rate 全表 + c4 391k 行评分体系审计：
  `results/grid5_report/k5/`（格子判断基本成立、top 端排序是噪声）
- DR 协议讨论 → Ben 决定 grid5-v2 全确定性 map（无扰动、champion 阶段 DR 后置）
  ——记录在 `dr_protocol_discussion_2026-08-26.md`，本轮五机已重启

## 5. 待定 / 下一步

- style_ref/ 下全部是风格样张（旧数据），不入正式报告
- 阈值规则（M1–M4）、T-slip 口径、foot clearance 基线归零：待 Ben 定
- grid5-v2 数据到齐后：同批脚本 `--round grid5` 直接出正式图 +
  REPORT/INDEX 再生成
