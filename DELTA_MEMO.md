# NCSA Delta 上机备忘录 — GRID-5 C2

> 建立于 2026-09-01。记录:为什么这么搭、有哪些规矩、现在到哪一步了。
> 姊妹文档:`../PenguMujoco_psc/PSC_MEMO.md`(C1 在 PSC Bridges-2,已停摆)。
> 下次隔了几周回来,先读这份。

---

## 1. Context:这是在干什么

把 GRID-5 的 **config C2**(`kappa=0.0, COM=1.20`)放到 NCSA Delta 上跑。
C2 在 GRID-5 里**一行都没跑过**,所以是 36 个切面全量:`hip_phi = 0,10,…,350`。

起因:PSC 上的 C1 战役预算耗尽(372 SU 起步,跑完 2 个切面后只剩约 58 SU),
而实验室在 Delta 上有一个现成的 CPU allocation。

**结论先说:Delta 的额度同样严重不足,但这是实验室自己的账户,余额可以用完。** 见第 4 节和第 7 节。

---

## 2. 东西在哪

### 本地(Mac)

```
~/Documents/CMU/Robomechanics Lab/
├── PenguMujoco/          ← 线上仓库，笔记本机队在跑 c8，【只读，不碰】
├── PenguMujoco_psc/      ← C1 on PSC Bridges-2（停摆）
└── PenguMujoco_delta/    ← 本树。Delta 全部工作在这里
    ├── DELTA_ISOLATED_TREE   隔离哨兵，说明复制了什么、故意没复制什么
    ├── DELTA_MEMO.md         本文件
    ├── models/pengu1_31/     模型（逐字节副本）
    ├── pengu_mujoco/grid5/   5 个源文件（逐字节副本）
    └── delta/                新写的脚本
```

三棵树三个哨兵。Delta 的作业物理上写不进 PSC 树或线上仓库,反之亦然。

### Delta

```
/projects/beht/bgu/          持久，项目配额 500G（建树时只用了 304K）
├── mjx_venv/                mujoco 3.8.1 + numpy + matplotlib
└── pengu/                   与 PenguMujoco_delta/ 一致
    ├── delta/state/         campaign 状态：budget/unit/costs/jobs/STOP
    └── pengu_mujoco/results/gait_sweep/   产出的 CSV

/u/bgu                       家目录，另一个文件系统，配额 100G，别放数据
/work/nvme/beht/bgu          500G NVMe，更快，但清理策略未确认
/work/hdd/beht/bgu           1T
```

### 为什么代码在 git 仓库之外

`grid5_sweep.py:39` 的 `_ROOT` 由自身 `__file__` 推导,**输出路径自动跟着代码树走**。
副本放在仓库外 ⇒ 物理上写不进线上 `results/gait_sweep/`,也不会被机队的复活器
(`run_machine.sh` 每 5 分钟轮询)捡走,更不会被误提交。**隔离靠结构,不靠小心。**

---

## 3. 规矩

1. **访问 Delta 只走名为 `delta` 的 tmux 会话**,禁止从 Bash 直接 ssh。已写进 `~/CLAUDE.md`。
   会话断了就**停下来重开**,不退回 ssh。
2. **花额度的动作**(`sbatch`/`srun`/`salloc`)提交前必须明确征得同意。
   只读命令(`squeue`/`sacct`/`accounts`/`quota`/`sinfo`)不用问,但同样走 tmux。
3. **线上仓库 `PenguMujoco/` 一律只读。** 机队 2026-08-31 跑完 c3,现在在跑 c8;
   复活器会把改动捡走并静默追加进同一个 CSV,而 `check_manifest` 只查
   protocol/config/K/mujoco 版本,**察觉不到指标代码变了**。
4. **账户是实验室共用的**(`/projects/beht` 下 15 个用户目录)。Ben 已定:余额可以全部用掉。
   但别人也在同时花 —— 所以 `burn` 的余额永远取「`accounts` 实时值」和「基线 − sacct 累计」
   里的**较小值**,高估余额等于花掉别人正要用的额度。

---

## 4. 实测数字

### 计费 —— **1 SU = 1 core-hour**

来源:NCSA Delta 用户手册 Job Accounting ——
"a node-exclusive job that runs on a compute node for one hour will be charged
128 SUs (128 cores × 1 hour)"。

✅ **已实测确认**(作业 21683493,2026-09-01):余额 **431 → 429**,正好 −2,
与 sacct 的 `AllocCPUS×Elapsed = 8 × 912s = 2.0 core-hours` 吻合;
node-hour 读数是 0.253,不匹配。**文档与实测一致。**

附带结论:**Delta 的 `accounts` 不像 PSC 的 `projects` 那样滞后** —— 作业结束几分钟内就更新了。
但守卫仍然以 sacct 为准、`accounts` 只用于对账:滞后行为可能随负载变化,而少算就会超支。

| | |
|---|---|
| 账户 | `beht-delta-cpu`,实验室共用(15 个用户目录 / 10 个 SLURM 关联),ACCESS Explore 档 |
| 余额(2026-09-01) | **431 / 已存 1142 Hours**(711 由其他成员花掉) |
| GPU | **无** GPU allocation ⇒ MJX 路线在本账户上跑不了 |
| 本地记账 | `sacct -X -o AllocCPUS,ElapsedRaw`(不滞后);`accounts` 会滞后 |

### 硬件与队列

| | |
|---|---|
| CPU 节点 | AMD EPYC **7763**(Milan)2×64 = 128 核 / 256G,136 个节点 |
| `cpu` 分区 | MaxTime **2 天**;**DefaultTime 只有 30 分钟** —— 忘写 `--time` 会超时被杀且照样扣费 |
| 计费权重 | `TRESBillingWeights=CPU=1000,Mem=512G` + `PriorityFlags=MAX_TRES`;`DefMemPerCPU=1000` ⇒ CPU 项主导 |
| 传输 | **`rsync` / `scp` 都有**(Bridges-2 两个都没有)⇒ 不需要 `tar \| ssh` + 手工 sha256 |
| Python | 系统 3.9.21;`module load python/3.13.5-gcc13.3.1`;**无 anaconda3** |
| 登录节点外网 | 通(pypi 200)⇒ pip 在登录节点做 |
| 其他 | `seff` 有;交互分区限 2 排队 / 1 运行 |

### 规模(继承自 PSC 的 C1 实测,对 C2 只是**上界**)

| | trial 数 | PSC C1 实测机时 |
|---|---|---|
| 1 个 hip_phi 切面 | 115,200 | **152.6 / 159.1 SU**(取 156) |
| C2 全量(36 切面) | 4,147,200 | **≈ 5,600 SU** |

**为什么只是上界**:C1 的 `com=1.05` 是十个 config 里最稳的,trial 最少提前摔倒 ⇒
跑满 21,000 步的比例最高 ⇒ 每行最贵。C2 的 `com=1.20` 更容易摔,应该更便宜。
加上 EPYC 7763 比 7742 单核快约 10–20%。**两项都指向更便宜,但都没测。**

### 标定实测(作业 21683493,8 核 × 912s,花 2 core-hours)

| | Delta C2 @ 8 核 | PSC C1 @ 8 核 | 比值 |
|---|---|---|---|
| rows/hr/core | **919** | 904 | **1.02×** |
| 每切面(115,200 行) | **125.4 core-hours** | 127.4(按 8 核率折算) | — |
| 36 切面全量 | **4,515 core-hours** | — | 余额的 **10.5 倍** |

⚠️ **一个被推翻的预期。** 建树时写着「C2 的 com=1.20 更容易摔 ⇒ 应该比 C1 便宜」,
外加「EPYC 7763 比 7742 快 10–20%」。实测 **1.02×** —— 两个效应加起来在噪声里。
教训与 PSC 那两次一样:**方向听起来对的推理,幅度可以完全不成立。**

因此 125.4 < 156 的原因**不是** C2 更便宜,而是 **8 核比 128 核每核效率高**
(PSC 上同样是 904 vs 754)。这反过来成了 burn 用少核的实测依据。

### 第一个真实切面(作业 21683962,phi=0,32 核)

**115,200 行 / 13,682 秒 / 121.6 core-hours**,gzip 后 19.7MB → 5.4MB(3.6×)。

| 口径 | rows/hr/core | 说明 |
|---|---|---|
| 8 核标定(网格起始区) | 919 | 采样窗口只覆盖网格开头 |
| 32 核(同一起始区,头 2 小时) | ~880 | **同口径比较 ⇒ 32 核约损失 4%**(PSC 128 核是 17%) |
| 32 核(整个切面均值) | **947** | 与 919 **不可直接比较** —— 含了后段便宜的区域 |

⚠️ **切面内部速率差异很大**:前两小时 ~880,第三小时冲到 **1,260**(+43%)。
原因是摔倒的 trial 提前结束,扫描推进到更容易摔的参数区就变便宜。
⇒ **每切面成本会因 phi 而异**,121.6 是 phi=0 的实测,不是所有切面的常数。

**32 核的选择被验证了**:同口径只损失 4%,远好于满节点的 17%,而墙钟(13.4h)远未触及 2 天上限。

### 账对不上的地方

余额 431,一个切面约 156 ⇒ **只够 2.8 个切面(36 个里的 8%)**。
全量 C2 需要余额的 **13 倍**。这不是意外,和 PSC 一样:
**整套设计就是为了在预算严重不足的前提下安全推进。**

---

## 5. 从 PSC 继承的坑(已在 Delta 的脚本里防住)

| 坑 | Delta 上的状态 | 防法 |
|---|---|---|
| **余额命令滞后** | PSC 的 `projects` 滞后 1–2 小时;Delta 的 `accounts` 未测 | 实时记账一律用 `sacct`;`accounts` 只用于对账 |
| **SLURM 把脚本复制到 spool 目录** | Delta 一样 | `.slurm` 里用 `SLURM_SUBMIT_DIR` + 哨兵检查 |
| **默认 `--time` 太短** | PSC 是 1 小时,**Delta 只有 30 分钟** | 两个 `.slurm` 都显式写 `--time` |
| **exit 0 ≠ 崩溃** | 同 | `c2_node.sh` 用 `wait $pid` 取退出码,0 就标 `done` |
| **`pip install` 不钉版本** | 同 | `pip install "mujoco==3.8.1"` + 末尾 assert |
| **matplotlib 是硬依赖** | 同 | `gait_sweep.py:30-32` 模块级 import,装环境时别漏 |
| **$HOME 与项目空间是两个文件系统** | Delta 是 `/u/bgu` vs `/projects/beht` | venv 和数据都放 `/projects/beht/bgu` |
| **核越多,每 core-hour 出的行越少** | PSC:8 核 904 vs 128 核 754 rows/hr/core | 按 core-hour 计费 ⇒ `burn` 默认 **32 核**而非 128;墙钟不是瓶颈,额度才是(第 7 节) |
| **没有 rsync/scp** | **Delta 有** | fetch 直接用 rsync,省掉手工 sha256 |

### 关于估算纪律(PSC 最贵的一课)

PSC 上两个估计被实测推翻:"指标层优化提速 15–55 倍"实际天花板 2–3 倍;
"EPYC 比 M2 Max 慢 1.5–2 倍"实际 2.2 倍。错因都是**拿文档当基准而不是实测**。

**规矩:先花 2 SU 标定,再规划。** Delta 上同样:`c2_calib.slurm` 先跑,
再决定要不要提交任何 128 核的作业。

---

## 6. 现在到哪一步了(2026-09-01)

| 项 | 状态 |
|---|---|
| 登录 / tmux 通道 | ✅ `delta` 会话,dt-login03 |
| 环境勘察 | ✅ 分区/文件系统/工具/Python/计费权重 全部实测 |
| 隔离树 | ✅ `PenguMujoco_delta/`,models + grid5 与 `_psc` 逐字节一致 |
| 脚本移植 | ✅ 9 个文件(含 burn 模式);本地语法检查 + 冒烟 + 哨兵拒绝 + burn 定额四个分支全过 |
| 代码同步到 Delta | ⬜ 待 Ben 在自己终端 rsync |
| venv | ⬜ |
| **标定** | ✅ 作业 21683493,2 core-hours,919 rows/hr/core |
| **计费单位** | ✅ 实测确认 core-hour(431→429) |
| **正式跑** | ✅ 作业 **21683962**,COMPLETED (0:0),13:14:52 × 32 核 = **424 core-hours** |
| 切面 | ✅ **3 完整 + 1 部分**(phi=0/10/20 各 115,200 行;phi=30 停在 51,755)|
| 余额 | ✅ **431 → 5**(烧掉 426 = 标定 2 + burn 424)|
| 数据回传 | ✅ **管道已验证**:8 文件 19,349,389 字节,两端聚合 sha256 `eea886dd…` 一致 |
| 数据校验 | ✅ `merge_phi.py`:**397,355 行 = 397,355 唯一键**,32 shard 并发追加零重复 |

### 关于账户归属(2026-09-01 已定)

项目名 **"training quadrupedal locomotion"**,Org = **`delta.explore`**
—— ACCESS **Explore** 档(最小的一档,所以只有 1142 core-hours),
`/projects/beht` 下 15 个用户目录全是本实验室的人。**不是 CMU 全校资源,是实验室的。**
Ben 的决定:**按实验室共用处理,余额可以全部用掉。**
`delta/state/reserve.txt = 0` 就是这个决定的落地。

---

## 6b. 第一次 burn 的完整结果(作业 21683962)

| phi | 行数 | 秒 | core-hours | 结果 |
|---|---|---|---|---|
| 0 | 115,200 | 13,682 | **121.6** | rc=0 完成 |
| 10 | 115,200 | 13,198 | **117.3** | rc=0 完成 |
| 20 | 115,200 | 14,283 | **127.0** | rc=0 完成 |
| 30 | 51,755 | 6,518 | **57.9** | **rc=3 到点停下**(可续)|
| | **397,355 行** | 47,681 | **423.8** | = **3.45 / 36 个切面** |

**deadline 收手完全按设计工作**:13:07:07 干净停下(作业时限 13:16,提前 9 分钟),
打包完成,`exit 0`。SLURM 记的 elapsed 13:14:52 × 32 = 424.0 core-hours,
与脚本自记的 423.8 差 0.05%。余额 431 → **5**。

**每切面成本 121.6 / 117.3 / 127.0,离散度 8%** —— 因 phi 而异,不是常数。
取 **122 core-hours/切面** 作为后续规划值(PSC C1 在 128 核上是 156,低 22%,
差别主要来自 32 核 vs 128 核的每核效率,不是 C2 本身更便宜)。

**`accounts` 不滞后**,两次验证(标定 431→429;burn 后 →5)。与 PSC 的 `projects`
滞后 1–2 小时不同。但守卫仍以 sacct 为准 —— 少算会超支,多算只会早停。

---

## 6c. 本轮新踩的坑

| 坑 | 现象 | 处理 |
|---|---|---|
| **登录节点会主动断开 ssh** | `Connection closed by remote host`,`delta` pane 退回本机 zsh。作业在 SLURM 上**不受影响**,但监控盲了 6 小时 | 见下 |
| **看门狗只 grep 正常标记 ⇒ 断线 = 静默** | 只匹配 `BURN_NEWS/BURN_END/FAILED`。ssh 一断,这些字样再也不出现,而"没有新行"和"一切正常"长得完全一样 | **必须加存活判据**:远端每次循环都打时间戳,监听端检测「超过 N 分钟无新行」就报警;或直接查 `tmux display-message -p '#{pane_current_command}'` 是否还是 `ssh` |
| **rc=3 分支不打包** | `fetch` 只拉 `*.csv.gz`,而部分切面不生成 `.gz` —— 对一个**设计上必然停在部分切面**的作业,等于每次都漏掉最后那批行 | 已修:rc=3 也 `gzip -kf` |
| **`du -h` 在 Lustre 上报分配块** | 日志里打成 `(512)`,实际 5.4MB | 已修:改用 `stat -c %s` |
| **移植时漏了第二个硬编码常量** | `merge_phi.py` 打印 `4 of 34 (phi 20..350)` —— 那是 PSC 上 C1 的范围。当初把它归进「A 类:逐字节复制,只改 glob」是错的 | 已修:`36 (phi 0..350)`,常量提到文件头 |
| **部分切面被报成 PROBLEM** | 唯一一份数据完整性报告在每次 burn 后都「狼来了」,真出重复键反而不显眼 | 已修:`INCOMPLETE(可续)` 与 `PROBLEM(真损坏)` 分开 |

---

### 剩下最要紧的两件事

1. ~~把 397,355 行拉回本机并验证管道~~ ✅ **已完成并校验**(2026-09-01)。
   数据在 `pengu_mujoco/results/gait_sweep/`;合并成单文件:
   `python delta/merge_phi.py --out c2_merged.csv`
2. **拿实测数字去要额度** —— 剩余 32.55 个切面 × 122 = **约 3,970 core-hours**。
   这个数有 4 个切面的实测背书,比任何估算都硬。
   (整个 GRID-5 十个 config = 36 × 10 × 122 ≈ **43,900 core-hours**。)

---

## 7. 「烧到底」模式(burn)

额度只够 2–3 个切面,所以战役的形态不是「36 个作业」,而是**一个跑到没钱为止的作业**。

```bash
bash delta/c2_ctl.sh burn        # 从实时余额算出方案，只打印，不提交
bash delta/c2_ctl.sh burn --go   # 真的提交
```

### 三个设计要点

**1. `--time` 就是预算,不是安全余量。**
Delta 按 core-hour × 实际 elapsed 计费,所以 N 核 × T 小时 = 恰好 N·T core-hours。
把 `--time` 定成 `余额 / 核数`,作业就在额度耗尽的那一刻停 —— 而不是留下一个
「不够跑完一个切面」的零头永远花不掉。这和平常「多申请点时间反正不扣」的直觉相反。

**2. 余额取两个来源里的较小值。**
`accounts` 权威但滞后(PSC 的对应命令滞后 1–2 小时);「建树时的基线 − sacct 累计」
不滞后但只看得见自己的作业(账户是 15 个人共用的,别人也在花)。高估余额=花别人的钱,
低估只是少花一点,而且**没跑完的切面下次能续**。

**3. 核数默认 32,不是 128。**
按 core-hour 计费时,重要的是**每 core-hour 出多少行**,不是跑得多快。
PSC 实测 8 核 904 rows/hr/core、128 核 754 —— 满节点因内存带宽和文件系统竞争损失约 17%。
而墙钟根本不紧张(431 core-hours 在 32 核上是 13.5 小时,分区上限 2 天)。
⚠️ **这是判断,不是实测**:8 核和 128 核之间的曲线在 Delta 上没测过,邻居作业的干扰也未知。
改 `delta/state/cores.txt`。

### 中途被砍会损失什么

**最多每个 shard 一行。** 行是逐行 flush 的(`grid5_sweep.py:298`),续跑按参数六元组去重
(`gs._load_done()`)。所以「烧到零」在这个负载上是安全的 —— 换成一个只在结尾出结果的
负载就完全不成立。

`c2_burn.sh` 会在作业时限前 **10 分钟**主动收手(`MARGIN=600`),留出 shard 退出 + gzip 的时间。
被 SLURM 硬杀的作业照样扣费,却跑不到打包那一步。

退出码:`0` 切面完成 · `3` 到点停下(部分完成,**不是错误**)· 其他 = 真故障,不继续下一个切面。

---

## 8. 常用命令

```bash
# ---- 在 Delta（tmux delta 会话）----
cd /projects/beht/bgu/pengu
bash delta/c2_ctl.sh status       # 进度 + 花费 + 还能跑几个
bash delta/c2_ctl.sh unit         # 计费单位：文档说法 vs 实际观测
bash delta/c2_ctl.sh burn         # 烧到底的方案（只打印）
bash delta/c2_ctl.sh burn --go    # 提交
bash delta/c2_ctl.sh start 0      # 或者：一次一个切面
bash delta/c2_ctl.sh stop         # 当前作业跑完就停
bash delta/c2_ctl.sh fetch        # 打印在 Mac 上该敲的 rsync
squeue -u bgu
sacct -X -j <id> -o JobID,AllocCPUS,ElapsedRaw,Elapsed,State
accounts                          # 滞后，别用来做实时决策

# ---- 在 Mac ----
# 推代码
rsync -av --exclude='.DS_Store' --exclude='__pycache__' \
  "$HOME/Documents/CMU/Robomechanics Lab/PenguMujoco_delta/" \
  bgu@login.delta.ncsa.illinois.edu:/projects/beht/bgu/pengu/

# 拉数据（bash delta/c2_ctl.sh fetch 会打印完整命令）
```

## 9. 文件

- `DELTA_ISOLATED_TREE` — 复制了什么、故意没复制什么
- `delta/c2_driver.py` — patch `HIP_PHIS`/`TAG` 后调用未改动的 `grid5_sweep.main()`
- `delta/c2_node.sh` — 一个切面 N 个 shard + 按退出码判断的重启 + deadline 收手
- `delta/c2_burn.sh` — 一个作业内跨切面连跑,到点停(**烧到底的核心**)
- `delta/c2_burn.slurm` / `c2_phi.slurm` / `c2_calib.slurm` — 三种作业
- `delta/c2_ctl.sh` — 控制器 + 预算守卫 + `burn` 定额 + `unit` 对账
- `delta/merge_phi.py` — 合并 + 校验

### 还没验证的

- `accounts` 到底滞后多久
- 计算节点能不能 `sbatch`(自动链条依赖它)
- `/work/nvme` 的清理策略,以及它能不能拿回 PSC 上因 Lustre 竞争损失的 17%
- 128 shard 在 Delta 上的扩展性(PSC 上 128 核 / 8 核 = 0.83)
