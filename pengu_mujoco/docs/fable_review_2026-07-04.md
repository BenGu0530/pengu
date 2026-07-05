# Fable review — 2026-07-04

Pengu 代码审阅 + 实验方向。产出两份:**A 代码整理**、**B 实验方向**。
证据均带 `file:line`。**PROPOSE not DISPOSE** —— 全是建议;改 gait/reward/model 或启杀 sweep 需 Ben 批准。

---

## 0. 现状快照(写文档时的 live 状态)

- **fine3c 6-DOF 网格正在跑**:16 个 `physics/gait_sweep.py` 进程,CSV `results/gait_sweep/sweep_v3_p25_fine3c_*.csv` 17:49 仍在写,~**74% 完成**(2.94M / 3.97M 行)。**别碰、别杀。**
- fine3c AXES([gait_sweep.py:78-85](../physics/gait_sweep.py#L78-L85)):`freq 1.00–1.50@0.01`(51)× `hip_phi 0–350@10°`(36)× `leg_amp{95..115}`(5)× `hip_amp{16..22}`(4)× `torso_amp{12,16,20}`(3)× `torso_phi 0–350@10°`(36)= 3.97M。
- ⭐ 注册获胜步态 `fine2_best_netfwd`(**penguV3**):freq1.59 / hip_phi180 / leg110 / hip20 / torso20 / torso_phi0 / hip_off30;net_fwd 0.226 m/s,μ_req p95 0.55。
- 论文核心实验 `physics/friction_study.py` = **penguV2 硬编码**,自带种子 CMA 优化到匹配速度,**未接入注册 winner**(下详)。

---

## A. 代码整理(按优先级)

### A1 · 🔴 friction_study 的步态来源与 winner / model 脱节(影响论文可复现性)
- `friction_study.py` 用**自己的硬编码种子** `leg_amp=70, hip_amp=8, torso_amp=12, freq=1.4, hip_phi=250, torso_phi=150`([:84-85](../physics/friction_study.py#L84-L85)),对 `leg/hip/freq(+torso_phi)` 做 CMA 优化到匹配速度 `V_TARGET=0.08`([:61](../physics/friction_study.py#L61))。**它不读 `BEST_GAITS.md` 的 winner,也没有注入接口**(`main()` 唯一 CLI 参数是 `maxfev`)。
- 且 `friction_study` **硬编码 penguV2**([:39](../physics/friction_study.py#L39)),而 winner 是在 **penguV3** 上调的(25° 动态 pitch)—— 即便把 winner 值塞进去也非同一模型。
- 另:`PHASE_OFFSET_A/B` 被强制置 0([:136](../physics/friction_study.py#L136)),外部 winner 若依赖非零腿相位偏移会被静默丢弃。
- **这不是纯清理,是实验设计岔路** → 见 B1 / C1,需 Ben 拍板"论文用 v2 还是 v3、friction_study 该自优化还是跑注册 winner"。

### A2 · 🔴 μ stance-gate 阈值口径:是不同用途还是 bug?
- `gait_sweep.py`:`F_HI=4N / F_LO=1N`([:91-92](../physics/gait_sweep.py#L91-L92))—— Schmitt 触发,gate **footfall 接触状态机 / 步数**。
- `friction_study.py`:`FN_MIN=1N`([:42](../physics/friction_study.py#L42),用于 [:182](../physics/friction_study.py#L182))、`grf_friction_probe.py`:`FN_MIN=1N`—— gate **μ_req 采样**(哪些帧计入 `|Ft|/Fn`)。
- scout 结论:两者 **gate 的是不同量**(状态机 vs μ 采样),所以不完全是"同指标不同阈值"。**但** `gait_sweep.py` 的输出里**确实带 `mu_req_p95` 列**(CSV header 确认),它用的 gate 与 friction_study 不同 → **两套工具的 μ_req 数字不可直接比较**。
- **待确认(bug or 有意)**:gait_sweep 的 `mu_req_p95` 具体用 `F_HI=4` 还是 `F_LO=1` 采样?若论文要横向引用 μ_req,须统一口径或明确声明二者含义不同。→ 快速定点读 gait_sweep 的 mu_req 计算即可判定。

### A3 · 🟡 死代码 / 死 import(安全清理)
- `from optimize_gait import evaluate`([friction_study.py:36](../physics/friction_study.py#L36))**从不使用** —— 可删。
- **8 个顶层老一代 sweep 脚本**:`sweep_amp_freq.py` / `sweep_phase_freq.py` / `sweep_anchor_validation.py` / `sweep_freq_surface.py` / `slice_2d.py` / `friction_scan.py` / `actual_gait_plot.py` / `render_gait.py` —— 均被 `physics/*` 取代,仅彼此 + 两份 prompt 文档引用,**无 Jun-7 之后的 results 产物**。注:`physics/analyze_sweep.py` 仍读老 anchor-validation CSV,归档前留意其输入格式依赖。
- `backup_scripts/`(6 文件,其中 `sweep_freq_pengu.py:18` 硬编码另一台机器的绝对路径 `/home/rml2/...`,明确死码)+ `backups/snapshot_20260607_160102.tar.gz`(65MB 整仓快照)—— 建议移 `archive/` 或删。
- **动作**:先删死 import(零风险);老脚本 + backup 需 Ben 点头再归档/删。

### A4 · 🟡 FOOT_BODIES dict 复制 5 份(安全去重)
byte-identical 五处:[grf_friction_probe.py:31](../physics/grf_friction_probe.py#L31) / [analyze_gait.py:29](../physics/analyze_gait.py#L29) / [friction_study.py:41](../physics/friction_study.py#L41) / [gait_report.py:30](../physics/gait_report.py#L30) / [gait_sweep.py:43](../physics/gait_sweep.py#L43)。
→ 提到共享模块(如 `friction_utils.py` 或新 `physics/common.py`)单一来源。**非行为改动,安全**。注意保留 body-name 陷阱注释(`right_foot0080___fillet13` = **左**脚)。

### A5 · 🟡 硬编码模型路径 + 误导性 usage banner
- [analyze_gait.py:27](../physics/analyze_gait.py#L27) / [gait_report.py:28](../physics/gait_report.py#L28) 硬编码 `penguV3/scene.xml`,**但 usage docstring 写 `PENGU_MODEL=v3 python ...`** —— 该 env var 对这两个脚本**无效**,banner 误导。
- `friction_study.py:39` / `grf_friction_probe.py:29` 硬编码 v2;只有 `gait_sweep.py:40`(`gc.XML_PATH`)、`cma_search`、`physics`、`model_scan` 真正尊重 `PENGU_MODEL`。
- **安全动作**:先修 docstring(删掉误导的 `PENGU_MODEL` 提示)。让脚本真正读 `gc.XML_PATH` 是行为改动(friction_study 是**有意**锁 v2),单独提议、别顺手改。

### A6 · 🟢 doc/code drift(更新 `prompt.md` §9,低优先)
- **`analyze_gait.py` 无 `FN_MIN`/`F_HI` 符号**(grep 未命中),但 `prompt.md §5` + `fable_prompt.md` 都称它用 `FN_MIN=1` —— 实际它的 `mu_req=|Ft|/Fn`([:5](../physics/analyze_gait.py#L5), [:103](../physics/analyze_gait.py#L103))**似乎未 stance-gate 或用别的门限**。drift,需订正。
- `gait_config.py` **file 默认值**两次 scout 读数有出入(细读:`WALK_HIP_AMP=0`, `WALK_CRANK_AMP=30`, `WALK_TORSO_AMP=15`, `PHASE_A/B=45/45`, `T_TRANSITION=2.0`;[gait_config.py:40-66](../gait_config.py#L40))。因 sweep 全覆盖,影响低,但 `prompt.md §4` 的"leg 内建 A=0/B=180"是 **sweep-time** 值,非 file 默认,措辞应精确化。

### A7 · 🟢 gait_config 全局 setter 不复位(已知陷阱,结构性)
`set_walk_freq/hip_amp/crank_amp/torso_amp` + 直接赋值 `PHASE_OFFSET_*` 均**不保存/复位**([gait_config.py:199-240](../gait_config.py#L199))。只有 `grf_friction_probe` 复位了 `PHASE_OFFSET_E`。
→ 提议加 `snapshot()/restore()` 或 context-manager。行为相邻(改语义),低优先,提议为主。

---

## B. 实验方向(对齐 open issue #6 / #2 / #1)

### B1 · 🎯 定清论文核心实验的"步态来源 + 模型"(最高优先,收敛论文)
论文主张"penguin 步态在湿滑更稳",那 friction_study 必须跑在**论文真正的对象步态**上。当前存在岔路(见 A1)。两条自洽路线,**需 Ben 决定**(C1):
- **(a) 保持 friction_study 在 v2 上自优化到匹配速度**:三种 torso 模式在**同速**下比 μ_req,去掉"走快 vs 倾斜多"的混杂 —— 实验设计上其实更干净。**推荐**,但要确认这是论文叙事。
- **(b) 移植到 v3 + 注入 fine2 winner 作基座**:让摩擦结果落在注册 winner 上;需处理 v2→v3 pitch 约定 + `PHASE_OFFSET_A/B` 强制零的问题。
→ 决定后,"跑 upright/over_stance/over_swing × μ ladder × 摩擦锥/最小可行 μ" 的路线才算钉死。

### B2 · 🎯 质量分布变体 pipeline(**issue #6**,交付物 B 核心)
- `gait_config.py` **无质量参数**,COM/质量在 **model XML** 里 → #6 需 Onshape 导出 human-COM 变体(54–57% 站立身高 + mid + upper),再逐个跑 friction_study 出 **μ_req vs COM/leg 曲线**。
- **阻塞**:Onshape 导出(外部)。**可提前做的准备**(非行为、纯扩展):把 friction_study 从"单个硬编码 XML"改成**遍历一组 model XML**,输出 CSV 加 `com_frac` 列 —— 导出到位即插即用。这条准备工作可现在提议实施。
- issue #6 还含硬化项:XM430 力矩限、floor priority fix、裁到 5 driven actuator、复刻 25° pitch —— 对齐 issue 原文。
- **STOP-ask(C2)**:plan 只给了 human COM 54–57%,**penguin 质量分布目标未给**(默认=当前模型?)。

### B3 · 🎯 issue #2 / #1 其实已被 sweep 覆盖 —— 别重复开新扫描
- **#2(低频 1.0–1.5)**:**fine3c 就是这个低频扫描**(`freq 1.00–1.50`,gait_sweep.py:74-77 注释直接点名),**正在跑**。→ 跑完 `physics/heatmaps.py` 找低频高点,**与 1.5–2.0 highland 对比**(#2 要的"比较两个 band")即可收 —— 对比需要早期 fine1/fine2 的高频数据。**不需要新 sweep**。
- **#1(φ=±90)**:**fine2 已做过 wide-phase 重扫**(hip_phi 30–330 含 90/270,见 gait_sweep.py:71-73 注释),并找到 `hip_phi=180` 的 winner(`fine2_best_netfwd`,BEST_GAITS 明确写"fine1 只扫 250–300 漏了它")。→ **#1 基本可关**,确认 fine2 覆盖了 ±90 且 Ben 认可即可。
- **动作**:fine3c 完成后做 heatmaps + 双 band 对比 → 关 #2;核对 fine2 相位覆盖 → 关 #1。**当前一次只能跑一个 sweep,勿启新的。**

### B4 · 论文前要收敛的指标口径
- **foot roll-vs-pitch 单峰检验**:friction_study 现只出 `foot_roll_amp`/`foot_pitch_amp` 幅值 + `torso_stance_corr`;plan 要的是"foot motion **单峰**"(slippery vs 非 slippery)。需确认现指标真能体现单峰,否则补峰计数。
- **μ_req 跨工具可比性**:即 A2,论文引用 μ_req 前统一 stance-gate 或声明差异。
- **v2/v3 一致性**:fine2 winner 在 v2 上是否仍是好步态?(与 B1 绑定)。

---

## C. 需 Ben 拍板的 ground-truth 问题(别猜)

1. **C1(论文模型 + 步态来源)**:friction_study 用 **v2 自优化匹配速度(推荐,更干净)** 还是 **移植 v3 + 注入 fine2 winner**?这决定 A1/B1 的所有下游。
2. **C2(penguin 质量分布目标)**:plan 只给 human COM 54–57%;penguin 侧目标 = 当前模型原生分布,还是另有数值?(issue #6 前置)
3. **C3(μ 阈值本意)**:gait_sweep 的 `mu_req_p95` 与 friction_study 的 `FN_MIN=1` 口径不一致 —— 有意(不同用途)还是 bug?论文是否要求二者可比?
4. **C4(真机表面 μ)**:`friction_utils.SURFACES`(mocap 0.7 / acrylic 0.30 / uhmw 0.14 / ptfe 0.06)是否即论文最终表面集?Naomi 硬件侧 μ 未在 plan 给数值。

---

*证据来自 4 个 code-scout(friction_study 管线 / gait_config 结构 / 清理证据 / plan PDF)+ 定点自读 gait_sweep AXES、friction_study CMA 段。冲突项(friction_study 是否已 baked-in winner)由主读 [friction_study.py:83-138](../physics/friction_study.py#L83-L138) settle:未 baked-in。*
