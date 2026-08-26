# K=5 体检：champion 可信度与评分体系 —— GRID-5 改进素材

2026-08-26。数据全部来自已有 topup 产物（`physics/topup_k.py`，权威 K=5）：
`results/grid4_report/cN/topupK5.csv`（c1/c3/c4/c5/c6 的 per-μ pass-track top-20
+ robust top-50@μ0.1）与 `results/gait_sweep/sweep_grid4_c4_topupK5.csv.gz`
（c4 全部 pass>0 格，391,285 行）。c2 无 topup，缺口保留。本文只记录数字与
待讨论项，不冻结任何规则。完整表格：`results/grid5_report/k5/champ_k5_table.md`、
`k5_audit_c4.md`；图 `k5_inflation_c4.png`、`k5_gates_c4.png`。

## 1. Champion 的 K=5 pass rate（map T-speed #1，每 config × μ）

| cfg | μ=0.1 | μ=0.3 | μ=0.5 | μ=0.7 |
|---|---|---|---|---|
| c1 | 5/5 | 5/5 | 5/5 | **2/5** |
| c3 | 3/5 | 5/5 | 5/5 | **1/5** |
| c4 | 5/5 | 5/5 | 5/5 | 5/5 |
| c5 | 3/5 | **1/5** | 4/5 | **1/5** |
| c6 | 3/5 | **1/5** | 2/5 | **1/5** |

24 个 champion 里只有 **9 个 5/5**；6 个 ≤1/5（纯单抽签产物）。c4 是唯一
四个 μ 全 5/5 的 config。

## 2. 膨胀幅度（K5 mean / K1 值，champion）

跨度 0.99（c1@0.1，几乎无膨胀）到 **-0.00**（c3@0.7，K5 均值归零）。
最重的失真：c6@0.1 0.602→0.163（×0.27）、c5@0.3 0.425→0.069（×0.16）、
c1@0.7 0.260→0.045（×0.17）。top-20 整体的中位膨胀比：c4 稳（0.75–0.96）、
c3/c5/c6 在 μ≥0.3 处 0.31–0.62。

## 3. 评分体系哪里成立、哪里失真（c4 全量 391k 行）

- **格子的"能走"判断基本成立**：K1 top-20 里 K5 pass≥0.8 的有 13–19/20；
  全 passer 群体 pass=1.0 占 86%/82%/57%/58%（μ=0.1/0.3/0.5/0.7）。
- **top 端的"排序"不成立**：K1 champion 在 K5-mean 重排后掉到第
  8 / **265** / **954** / 213 名（μ=0.1/0.3/0.5/0.7）；K1 与 K5-mean 的
  top-20 集合交集仅 9 / 5 / **1** / 10 个。即：K1 选出的格子大多真能走，
  但 K1 值排出的名次在 top 端是噪声的顺序统计量。
- μ≥0.5 全面变脆：全群体 pass=1.0 从 ~85% 掉到 ~57%。

## 4. 阈值候选在数据上切在哪（`k5_gates_c4.png`）

对 c4 的 K1 top-20：
- `pass_rate ≥ 0.8`（4/5）：保 65–100%（μ=0.3 最狠，砍 35%）；≥1.0 再砍到 55–90%。
- `net_fwd_min > NET_MIN(0.05)`：保 80–95%，与 pass≥0.8 高度重合但不等价
  （min 曲线在 0.05–0.15 平缓下降，0.15 后断崖）。
- c3/c5/c6 的 top-20 同门槛会砍掉 50–100%（champ_k5_table.md 第二表）。

候选规则（供讨论，样例重排见 champ_k5_table.md 的 M1/M2/M4 列）：
- **M1** floor `pass_rate ≥ 0.8`，组内按 K5 mean 排 —— 复用 0.8 这个已有阈值语汇。
- **M2** 直接按 `net_fwd_min` 排，gate `min > NET_MIN` —— 零新常数，worst-case 语义。
- **M3** mean − λ·(mean − min) —— 聚合文件无 σ，只有此形式可用。
- **M4** M1 ∧ nbhd ≥ 0.8 —— seed 稳健 ∧ 空间稳健双保险；注意 freq 边缘格 nbhd=NaN
  会被连带排除（c4@0.3 champ 就在边缘），需先定 NaN 的处理。

## 5. 对 GRID-5 的建议清单（待 Ben 定夺，逐条可独立采纳）

1. **报告层**：champion/top-N 数字一律并标 `pass_rate@K5`、`net_fwd_min`、nbhd —— 
   单独一个 K1 速度数不再出现在任何表里（本轮 c6@0.1 的 0.602 即教训）。
2. **选择链**：topup（step 3）之后、确认 seed（step 4）之前，加一步
   "按选定规则（M1–M4 之一）重排"，champion 从重排后的名单里取。
   规则选哪条、阈值取多少 —— **未定，先讨论**。
3. **T-speed 资格**是否维持 pass-only：pass-track 专抓峰值、robust-track 专抓平台，
   两条并报（speed 图已按此实现）；还是给 T-speed 也加 nbhd floor。
4. c4 全 passer topup 的性价比高（一次 topup 换来整套标定）；GRID-5 是否
   对每个 config 至少全 passer topup 一个 μ 档，作为标定样本。
5. 手搓 trial 循环与 topup_k.py 结果不可互换（本 session 实测：同 cell 同 seed
   公式，手搓 3/5 vs 权威 5/5）——K=5 数字只认 topup_k.py 产物，写进 PLOT 约定。

## 5.5 topup 要挖多深（c4 全量数据实算）

K5-mean 口径的真 champion 在 K1 榜上的排名：μ=0.1 第 **15,448**、μ=0.3 第 20、
μ=0.5 第 **42,688**（passer 共 57,950）、μ=0.7 第 **34,832**。即：只 topup
K1-top-20 能打假（拆穿膨胀 champion）但不能寻真——c4@0.3 的
f1.99/φ280/off30（5/5, mean 0.535, min 0.500，φ280–290 成排平台）只有全量
topup 才露出来，并反超 c3@0.3 的 0.490/min 0.251（双格尖峰）。

Caveat：K5-max 本身仍是 10^5 量级样本上的顺序统计量，会再膨胀一次（比 K1 轻）；
champion 可信度需平台性（邻居同为 5/5）+ step 4 的 +50000 独立确认 seed 背书。

GRID-5 的深度选项（待定）：
- A 维持 top-20 topup（只打假）；
- B 全 passer topup ≈ 4×passer 数 ≈ 0.7–0.8 个 map 成本/config（能寻真）；
- C per-μ top 2,000–5,000（小时级，罩住 μ0.3 类，深藏者仍可能漏）；
- D 两段漏斗：宽 N 先 K=3，幸存者补 K=5——同预算挖更深。

## 6. 缺口

- c2 无任何 K=5（map 完整，`physics/topup_all.sh c2` 或 per-μ top-20 选择即可补）。
- c4 全量文件以 r=0 通过为条件（pass_rate 下限 0.2），对 champion 过滤无碍
  （champion 必然 r0 通过），但不能拿它估"全格子"的 seed 方差。
- +50000 独立确认 seed（step 4）在 sweep 侧还没有 runner。
