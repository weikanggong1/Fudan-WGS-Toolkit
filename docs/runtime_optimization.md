# 关联运行时优化

本次将 `guo` 的关联运行时改动整合到主分支，保留主分支已有的单样本预处理、数据准备和两类关联入口。输入、参数和输出结构沿用[连续表型](wgs_continuous.md)与[二分类表型](wgs_binary.md)说明，不增加科学参数或依赖。

## 改动与精度策略

| 改动 | 作用 | 保留的行为 |
| --- | --- | --- |
| 二分类 SPA 求根 | CUDA FP64 且一维 score 向量连续时，合并导数与求根控制计算，减少 Python 与设备之间的同步；支持基因型与病例概率的步长布局 | 原求根区间、收敛判据、尾概率和诊断字段；非连续 score 等情况沿用原参考分支 |
| CUDA 内存查询 | 每次从一个原生 allocator 快照读取当前 allocated / reserved | 实时 CUDA free、进程显存上限与缓存预算；计数不缓存 |
| 主存账本 | 同一租约只改变阶段标签时减少重复落盘 | 互斥锁、进程身份、预算变化时的持久写入及增长前的实时 PSS 检查 |
| CCT 条件计算 | CUDA 上含 1–4,096 个概率的小向量合并条件归约，全部概率都可使用常规公式时减少布尔索引 | 极小 P 的稳定分支、非有限值检查与边界处理 |
| 精度审计 | 用实际算子对象识别被审计操作 | 原 TF32 / FP32 / FP64 选择与审计范围 |

主存账本文件中的阶段标签对应最近一次预算、所有权或租约标签变化；活跃租约对象继续记录当前阶段。仅阶段变化不触发重复的 fsync。

`_cuda_memory` 在首次查询 CUDA 时才导入 Torch，因此单独运行预处理的环境仍可使用仅依赖标准库的主存资源检查。关联继续使用已有 TF32 / FP32 路线，SPA 保持 FP64，没有引入 float16。

## 调用与输入输出

用户继续调用原有公开函数或命令行入口。完整参数、表型与协变量各自的 CSV 格式、prepared 遗传目录、模型与结果目录结构，见两类关联说明。固定模型的内部性能对照使用相同输入、相同模型、相同完整遗传任务和原科学配置；输入路径、模型文件及逐位点结果留在私有证据中。

## 验证记录

2026-10-26 的整合基于主分支 `f9d2fdc` 与 `guo` 的 `4805a879`。主分支增加的预处理实现保留；同时修复了优化辅助模块提前导入 Torch 对独立预处理环境的影响。

| 检查 | 结果 |
| --- | --- |
| 主分支完整 CPU 回归 | 3,692 通过，224 跳过 |
| 整合候选完整 CPU 回归 | 3,696 通过，318 跳过 |
| CUDA 与导入兼容性专项 | 157 通过，0 跳过 |
| 无 Torch 环境的预处理资源检查 | 6 通过 |
| 真实 chr21、16 连续与 16 二分类完整前后对照 | 全部原定任务通过，最终 CSV 解压字节完全一致 |

专项覆盖 SPA 求根的连续与非连续布局、共享与逐列病例概率、接近支持边界的尾部、迭代上限，以及资源拒绝和诊断字段。CPU 回归中的跳过项保留原设备与环境条件，不算作通过。

### 真实 chr21 完整同输入对照（2026-10-27）

两版复用同一套原始 prepared 输入和已拟合零模型，覆盖各自全部原定遗传任务、16 个表型、individual / coding / noncoding / ncrna 与原有全部 mask。四个关联阶段及控制器均已实际成功退出；持久完成确认、最终 CSV 完整回读、独立输出比较和输入 / 模型 / 源码绑定均通过。

| 表型族 | main 关联 API（秒） | 整合版关联 API（秒） | 观测 API 耗时比 main / 整合版 | 最大 `abs(Δ(-log10 P))` |
| --- | ---: | ---: | ---: | ---: |
| 连续 ×16 | 13,089.07 | 12,009.99 | 1.090 | 0 |
| 二分类 ×16 | 13,746.05 | 9,137.90 | 1.504 | 0 |

两类表型的全部最终 CSV 解压字节完全一致。全部可比较概率满足严格门槛 `<0.001`，无超限值、无缺少原生 log P 的未决下溢零值，元数据、数值分类及结构检查均无差异；没有忽略失败列或用人为 floor 代替尾概率。

按用户指定，两版在不同 GPU 上运行，主机负载存在重叠。整合版两类表型的本次观测关联耗时均较短；表中的比值属于这一运行条件下的观测，尚未进行同一 GPU 串行隔离试验，不能据此分离纯代码优化的因果效果。科学输入、模型、mask 与原有精度策略保持一致。

### 计时范围与分步骤记录

关联 API 使用明确同步边界，包含流式输入、结果写出、writer finish、最终 CSV 审计与运行时清理。下面各阶段有嵌套，不能相加作为端到端时间；模型加载及验证包含在初始化中，未单独给出其独立耗时。

| 阶段（秒） | 连续 main | 连续整合版 | 二分类 main | 二分类整合版 |
| --- | ---: | ---: | ---: | ---: |
| 完整输入 SHA 校验 | 36.97 | 36.84 | 37.09 | 37.19 |
| 初始化，含模型加载及验证 | 45.37 | 43.76 | 56.31 | 50.71 |
| API 去除 writer finish 与最终审计 | 13,015.02 | 11,935.13 | 13,648.56 | 9,039.05 |
| writer finish 与最终 CSV 审计 | 74.05 | 74.86 | 97.49 | 98.85 |
| 独立持久输出检查 | 1.33 | 1.35 | 1.37 | 1.39 |
| worker 总时间，含清理 | 13,136.01 | 12,055.35 | 13,803.94 | 9,190.21 |
| 外层启动至实际退出、会话清空 | 13,140.97 | 12,060.97 | 13,804.98 | 9,195.55 |

两版结果之间的独立完整比较另耗时：连续 289.62 秒，二分类 471.75 秒，不属于任一版关联 API。原始预处理、数据准备与零模型拟合本次均未重跑，因此这些阶段不计入该对照。输入校验等阶段的未舍入时间和明确范围保存在[匿名验证汇总](../validation/runtime_optimization_20261027.anonymous.json)。

本次验证的是同输入、同固定模型的优化前后实现一致性；与原生 R 软件的比较沿用各功能说明中既有且单独标明范围的记录。

## 更新记录

- 2026-10-27：完成真实 chr21 两类各 16 表型的完整输出验收，补充关联及独立检查耗时；保留分卡观测条件。
- 2026-10-26：整合 Guo 的 SPA、allocator、CCT 和主存账本优化，修复无 Torch 预处理导入兼容性，完成 CPU / CUDA 回归。

## 来源

- [Guo 优化提交](https://github.com/weikanggong1/Fudan-WGS-Toolkit/commit/4805a8793e113c9efad25fa012dd3732730762b0)
- [整合前主分支](https://github.com/weikanggong1/Fudan-WGS-Toolkit/commit/f9d2fdcc3adf54c47c19ccd6892ec823ced39779)
- 官方软件、方法文献和既有精度比较沿用[二分类](wgs_binary.md)与[连续表型](wgs_continuous.md)说明中的来源。

## 可选的 CPU 帧读取与解码并行

`phewas_run.portable_reader(reference, spec=None)` 可在读取既有压缩六状态 CSR 容器时启动独立 CPU 解码进程。子进程调用原有帧 SHA 校验和整数解码，主进程仍按原请求消费；样本轴、变异筛选、模型、mask、GPU 数值精度和统计计算保持原逻辑。默认关闭，保留原同步解码与单线程压缩帧预读路径。

输入 `reference` 为容器目录字符串，或含下列字段的字典；字符串形式可通过染色体 `spec` 字典传递设置。

| 参数 | 意义与格式 |
| --- | --- |
| `container_directory` | 已有 CSR 容器目录，包含 manifest、帧索引、压缩数据和样本轴；不生成或改写原容器。 |
| `metadata_directory` | 对应的只读元数据目录，样本轴须与容器一致。 |
| `frame_prefetch_processes` | CPU 解码进程数，默认 `0`；启用时取整数 `1..32`。这是实际子进程数，不是 BLAS 线程设置。 |
| `frame_prefetch_memory_bytes` | 正整数。对待处理帧的解码临时空间及 mmap 交接数据作保守字节准入；不包括解释器基线和消费者已有 LRU，仍须另设整个进程树的主存预算。 |
| `frame_prefetch_directory` | 可写私有临时父目录。仅在其新建独立交接目录，不写入原数据容器；关闭时等待自有子进程并回收交接文件。 |

返回的 reader context 提供原有 `metadata` 和 `container`。`container.read_frame(frame_id)` 的整数数组、dtype、shape 和列顺序与原解码一致。`prefetch_frames(frame_ids)` 只调度帧号；顺序、重复或非连续请求均由消费者按需读取，队列同时受任务数和字节预算约束。子进程不初始化 CUDA，每个数值库线程上限为 1；GPU 关联保留在原消费者进程。

```python
from pathlib import Path
from fudan_wgs_toolkit.phewas_run import portable_reader

# 原数据保持只读；tmp_directory 是调用者提供的私有可写临时目录。
tmp_directory = Path('/path/to/private/tmp')
tmp_directory.mkdir(parents=True, exist_ok=True)
reader_reference = {
    'container_directory': '/path/to/existing/container',
    'metadata_directory': '/path/to/existing/metadata',
    'frame_prefetch_processes': 4,
    'frame_prefetch_memory_bytes': 512 * 1024**2,
    'frame_prefetch_directory': str(tmp_directory),
}
with portable_reader(reader_reference) as resources:
    container = resources['container']
    # 帧号来自本次真实请求，不能用表型或 CSV 行号替代。
    container.prefetch_frames([0, 1, 2, 3])
    frame = container.read_frame(0)
    print(frame['m'], frame['n'])  # 本帧变异数、原容器样本数
```

这些选项属于 reader 引用或染色体 `spec`，不是新增命令行参数。现有命令行关联入口仍为 `python -m fudan_wgs_toolkit.phewas_run --plan /path/to/private/plan.json --association-only`；只有将 reader 设置传入上述引用的调用路径才启用该进程池。

运行指标区分已启动进程、实际完成任务、被消费或丢弃的预取结果及子进程 CPU 时间，并保留出生身份和实际退出状态。创建多个进程不等于这些进程持续繁忙，解码并行也不保证消除 GPU 调度或写出阶段的瓶颈。该选项新增串并逐数组字节一致性、任意读取顺序、预算、错误传播和退出回收测试；完整真实数据的精度及速度结论须以对应新评估记录为准，不能沿用前述旧版本结果。

## 可选的 Single 跨帧批次合并

两类关联入口的 `single_cross_frame_batches` 为布尔参数，默认 `False`，沿用按物理帧计算 Single 的方式。显式设为 `True` 时，从既有遗传容器读取可信整数摘要，将有机会通过原 MAC 筛选的列按原顺序合成有界请求；每次请求最多包含 `min(individual_effective_block_size, individual_genotype_block_size)` 列。现有公开参数 `individual_effective_block_size` 默认为 1,024，可设为 128、256、512 等值调整合并批宽；公开构建器的原始请求上限 `individual_genotype_block_size` 为 1,024，因此调大 effective 值不会超过该上限。关闭合并时继续使用原始物理帧请求及 genotype 上限，effective 值不改变此路径的批次或 checkpoint 键。各表型继续使用自己的样本集合、MAC、填补、分数、方差和 SPA。输入 CSV、prepared 目录与最终 CSV 结构沿用原入口，准备好的遗传文件保持只读。

```python
from fudan_wgs_toolkit import run_WGS_continuous

# 输入为已有的 FID/IID 表型表、协变量表及 prepared 遗传目录。
# 本例显式启用 Single 合并；其他参数含义见连续表型说明。
association_report = run_WGS_continuous(
    phenotype_csv='/path/to/phenotypes.csv',
    covariate_csv='/path/to/covariates.csv',
    prepared_directory='/path/to/prepared',
    output_directory='/path/to/new_results',
    chromosomes=[21],
    analyses=('individual', 'coding', 'noncoding', 'ncrna'),
    device=['cuda:0'],
    cpu_threads=8,
    grm='none',
    single_cross_frame_batches=True,
    individual_effective_block_size=256,  # 合并请求最多 256 列；仍受原始 1,024 列上限约束。
)
print(association_report['completed'])
```

二分类入口 `run_WGS_all_binanry(...)` 使用同名参数。两类命令行均提供 `--single-cross-frame-batches` 与 `--no-single-cross-frame-batches`，省略时关闭：

```bash
run-WGS-continuous phenotypes.csv covariates.csv /path/to/prepared \
  --output-directory /path/to/new_results --chromosomes 21 \
  --grm none --single-cross-frame-batches --individual-effective-block-size 256
```

批次策略、实际合并批宽和选定变异顺序与 plan、配置和 checkpoint 身份绑定。新 plan 关闭该选项时沿用旧配置形状，缺少该键即表示 `False`。`resume=True` 保留原配置表示及其校验值，也接受已有的显式 `False` 表示；数据、模型、源码及其他配置仍须通过原校验。切换为 `True`、从 `True` 关闭或调整 `individual_effective_block_size` 时，须使用新输出目录；即使一小段数据未填满批次，也不会复用不同实际批宽的 checkpoint。内部 `--plan` 入口也会拒绝改变已有 plan 的显式开关。该选项只改变 Single 请求的组织，不改变 gene-based mask。启用后的完整精度与性能结果使用对应的新验证记录。


## 主机 PSS 文本解析

共享节点的主存准入在每次资源增长时，现场读取已核验任务根进程及其全部自有后代的 PSS。支持 `smaps_rollup` 的内核沿用该接口；旧内核使用完整 `smaps` 捕获，并逐个核验内存映射范围、字段名及 `Rss`、`Pss`、可选 `SwapPss` 计数。输入、返回指标和主存预算参数沿用原接口。

完整捕获后，块解析器通过 C 正则引擎检查每个映射及其字段语法，Python 汇总三个计数。单个快路径块最多处理 64 个字段；更长的合法块、语法异常或计数校验未通过时，将同一次捕获的文本交给原逐行解析器，保留原有返回值或错误原因。后续映射损坏会使整次 PSS 读取保持未知，已有前缀计数不会作为完整结果使用。

每次读取仍执行原有的缓冲区、EOF 和字节上限检查；进程出生身份、完整后代覆盖、任务 PSS 上限、共享主存余量及 lease 判定沿用原逻辑。该变化减少 CPU 文本解析开销，无新增调用参数，也不改变关联输入、统计计算或输出结构。


## 二分类 Single 的 SPA 临界值复核

跨帧合并改变矩阵宽度后，FP32 分数或 TF32 方差的微小舍入差异可能让 nominal P 穿过 0.05，从而改变是否调用 SPA。启用 `single_cross_frame_batches=True` 的二分类 Single 现在对 `abs(nominal_p - p_filter_cutoff) <= 2e-3 * p_filter_cutoff` 的位置执行原计算几何复核。此范围只触发重算；最终仍严格使用 `nominal_p < p_filter_cutoff`，不会扩大 SPA 的筛选阈值。

复核从本次任务的完整 QC/type 变异轴定位原物理请求，沿用原 `individual_genotype_block_size`、MAC、样本顺序、等位基因方向和缺失填补，并以完整原表型批次计算原 FP32 Score 与 TF32 方差。原本由极端尾数触发的整列表型 GEMV 回退也会执行。只有接近临界值的二分类单元更新 Score、方差及其后续 normal P、效应估计与 SPA 判定；连续表型和其余单元保持原路径。FP64 SPA 的求根、容差和回退规则沿用既有实现。

该过程一次只保留一个原请求的临时基因型，GPU 分配先执行现有显存准入；有界的主存索引、标记和元数据临时量并入当前 Single 输出租约。结果仍按当前任务的原变异顺序、ordinal 和 CSV 结构发布，不发布复核过程中从零开始的临时 ordinal。报告中的 `single_binary_threshold_recheck` 给出触发、原请求重算及恢复单元数；策略、阈值与原始请求上限参与 checkpoint 身份，旧源码或旧策略的块不能与本版混用。完整精度和耗时结论需以本版冻结源码的真实数据验证为准。


## 二分类 Single 的选定列 SPA 工作区

原生 PheWAS 调度器先完成原有的普通分数、方差与 SPA 选择，再按实际选中的列数预估 SPA 统计工作区。预估宽度为 `min(selected_count, 512)`；实际求解仍使用原来的至多 512 列批次。普通 P 的严格选择、变异顺序、FP32 Score、TF32 方差及 FP64 SPA 求根保持原实现。选中列较少时，无需按整个普通检验块预留 SPA 临时空间；全部列都选中时保留原完整工作区预估。

统计工作区、拟合源上传和拟合投影分别执行准入，已有的基因型与拟合状态仍计入实时 allocator。模型仓库的既有最低工作区、进程显存上限和安全余量继续生效；正在使用的拟合状态仍由 pin 保护。该变化不预占整卡，也不绕过实时 CUDA free 检查。

低层函数 `fudan_wgs_toolkit.binary.binary_single_phewas` 支持下列互斥的 SPA 状态来源；高层 WGS 入口沿用原调用和命令行参数，内置调度器自动使用选定列准入。

| 参数 | 默认值与格式 | 含义 |
| --- | --- | --- |
| `spa_model` | `None`，或已验证的 FP64 SPA 模型 | 提供与普通模型具有相同拟合来源和有序样本轴的现成 sidecar。 |
| `spa_acquire` | `None`，或零参数上下文工厂 | 保留原有惰性接口；有选中列时调用一次，返回同一拟合来源的 FP64 sidecar。 |
| `spa_acquire_selected` | `None`，或接收一个正整数的上下文工厂 | 接收本次完整的 `selected_count`，由已计算的严格选择向量取得；返回同一拟合来源的 FP64 sidecar。实际分块宽度继续由原 solver 管理。 |

这三个参数至多提供一个。全部列未选中时不调用惰性工厂；原零参数工厂的调用方式不变。工厂内部的异常直接传播，不通过捕获 `TypeError` 猜测签名或重复调用。结果中的概率、原生 log P、SPA 选择与求根诊断字段保持原结构。

临界值复核、选定列工作区和主机资源优化的完整精度及耗时须绑定同一版冻结源码验收。前文标明日期的历史对照保留其原版本和测量范围；新增版本结果在独立完整输出比较和测量链通过后记录，不以定点回归或短时利用率替代。

### 2026-10-30：完整同输入运行时验证

连续与二分类的完整关联输出均通过两套独立保留基线比较，所有可比较 abs(Δ(-log10 P)) 严格小于 0.001；最大值分别为 1.3504688e-5 和 3.3668930e-5。适用失败列、未知尾概率和源公式数值违规均为 0。关联 API 分别为 12270.656 秒和 5591.685 秒，模型拟合未重跑。

API 窗口内时间加权的自有 CPU 平均约 1.105 / 1.095 核，自有 GPU SM 利用率约 53.398% / 28.290%，整卡约 55.608% / 29.753%。自有 GPU SM 有效覆盖为 95.378% / 96.542%，缺测保持 unknown。设备满载目标未达到，共享运行未用于宣称独占加速。

完整范围、计时边界与测量覆盖见 [完整验证记录](benchmarks/runtime_validation_20261030.md)。
