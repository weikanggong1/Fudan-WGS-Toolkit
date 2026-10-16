# 合成 WGS 测试数据

## 1. 功能与流程

`generate_simulated_data` 直接生成可供 `run_WGS_all_binanry` 和 `run_WGS_continuous` 读取的 **prepared 遗传目录、表型 CSV 和协变量 CSV**。它用于不接触真实个体数据的速度优化、内存测试及同一合成输入上的数值回归。生成阶段只用 CPU，不创建 PLINK、GDS 或 aGDS 中间文件，也不启动关联分析。

默认 `preset="reference"`、`chromosomes="all"`、`variant_scale=1`。参考模式从私有校准文件读取各染色体的物理位点数、MAF / 缺失分布和匿名 mask M。当前已提取的完整参考规模为 **22 条常染色体、1,023,836,503 个物理位点、345,967 个样本**；每条染色体采用自己的计数，不按染色体长度或染色体数平均分配。345,967 是遗传数据的完整样本轴，不是某个表型剔除缺失后的拟合人数。

参考 mask 固定取一份官方 STAAR PheWAS 表型结果的 15 类 gene 文件，按染色体及匿名基因组提取 M；不会把多个表型的结果行重复相加。这里的 M 是该 mask 的合格变异数，已发布 mask 数是官方结果中有记录的基因与类别组合数；两者均不等于物理位点总数、表型数、遗传帧数或关联 CSV 行数。

真实的逐染色体计数、频谱、匿名 M 表及来源绑定留在私有环境。公共仓库提供生成方法、校准格式和完全人工的 `preset="small"`，没有参考文件时明确报错，不使用人工小规模作为参考默认值。

```mermaid
flowchart TD
    PROFILE["私有汇总校准 / 显式 small 人工预设"] --> CHECK["核对格式、计数和 mask 关系"]
    CHECK --> PLAN["按 n、位点规模、稀疏度预估内存与磁盘"]
    PLAN --> CALLS["逐位点构造人工等位基因调用\nMAF 保持，整数 MAC 随 n 缩放"]
    CHECK --> ANNOT["人工坐标、身份、注释和 mask 归属"]
    CALLS --> PREP["生产格式的六状态稀疏 prepared\n逐帧计数、压缩、SHA 和完成清单"]
    ANNOT --> PREP
    PREP --> PHENO["人工 dosage + 协变量\n二分类 / 连续表型与独立缺失"]
    PREP --> AUDIT["频谱及各 mask 目标 M / 实际 M 核对"]
    PHENO --> REPORT["simulation.json + 两类调用参数"]
    AUDIT --> REPORT
    REPORT --> BINARY["run_WGS_all_binanry"]
    REPORT --> CONT["run_WGS_continuous"]
```

## 2. Python 调用

主页 Conda 环境已包含所需依赖。下面的人工小规模例子生成两条染色体、每条 1,024 个位点及两类表型，随后分别运行四类分析。`preset="small"` 必须显式指定；它的频谱和 mask 数没有真实队列含义。

```python
from pathlib import Path
from fudan_wgs_toolkit import (
    generate_simulated_data,
    run_WGS_all_binanry,
    run_WGS_continuous,
)


def main():
    # 必须是新目录；产物放在公共仓库外。
    simulated_data_directory = Path("work/simulated_small")
    simulation_report = generate_simulated_data(
        output_directory=simulated_data_directory,
        preset="small",                 # 完全人工的格式与接口测试预设。
        n=1000,                         # 样本数；改变 n 时保持目标 MAF。
        chromosomes=[21, 22],
        variant_scale=1.0,               # small 中每条 1,024 个位点。
        phenotype_type="both",
        n_binary_traits=2,
        n_continuous_traits=2,
        phenotype_missing_rate=0.02,     # 各表型独立生成缺失。
        covariate_missing_rate=0.0,
        seed=1729,
        chunk_size=256,
        cpu_threads=2,
        memory_limit_gib=8,              # 生成阶段主存估算预算。
    )
    print(simulation_report["prepared_directory"])
    print(simulation_report["mask_counts_exact"])

    # 返回参数已经包含表型、协变量、prepared、四类分析和独立输出目录。
    binary_analysis_arguments = dict(simulation_report["binary_kwargs"])
    binary_analysis_arguments.update(
        device=["cuda:0"],
        memory_limit_gib=20,             # 关联阶段每卡的准入预算。
        host_memory_limit_gib=8,         # 关联阶段任务的节点主存预算。
        resume=False,
    )
    binary_association_report = run_WGS_all_binanry(**binary_analysis_arguments)
    print(binary_association_report["completed"])

    continuous_analysis_arguments = dict(simulation_report["continuous_kwargs"])
    continuous_analysis_arguments.update(
        device=["cuda:0"],
        memory_limit_gib=20,
        host_memory_limit_gib=8,
        resume=False,
    )
    continuous_association_report = run_WGS_continuous(**continuous_analysis_arguments)
    print(continuous_association_report["completed"])


if __name__ == "__main__":
    main()
```

两类返回参数均包含 `grm="none"`、`gene_variant_type="variant"` 和 `analyses=("individual", "coding", "noncoding", "ncrna")`。`variant` 保留人工 Indel，以覆盖 PTV；成熟关联内核在该模式下不使用数值功能注释权重。需要测试11个数值注释通道的加权计算时，将返回参数改为 `gene_variant_type="SNV"`，同时使用新的关联输出目录；SNV 筛选会改变 PTV 等 mask 的资格与 M。这两种负载应分别记录，不改变现有关联行为。多 GPU 按完整染色体分配，一条染色体不会自动拆给多张卡。

二分类返回 `impute_missing=False`，连续返回 `rm_nan=False`，分别保留每列表型的 available 样本。协变量缺失始终剔除。需要共同样本轴时可把 `phenotype_missing_rate=0`；需要测试其他标签策略时，在新的关联输出目录中分别设置 `impute_missing=True` 或 `rm_nan=True`。这些选择会影响关联阶段的 MAC、MAF 和 mask 资格。

### 参考规模生成

私有参考文件配置完成后，不需要再次访问原始个体数据。下面保留所有染色体的物理位点数和 mask 目标，只将样本数改为 10,000：

```python
from fudan_wgs_toolkit import generate_simulated_data

if __name__ == "__main__":
    reference_simulation_report = generate_simulated_data(
        output_directory="work/simulated_reference_n10000",
        calibration_profile="/private/calibration/simulation_profile.json",
        n=10000,
        phenotype_type="binary",
        n_binary_traits=16,
        cpu_threads=8,
        memory_limit_gib=8,
    )
    print(reference_simulation_report["frequency_counts_exact"])
    print(reference_simulation_report["mask_counts_exact"])
```

示例路径由使用者替换；参考文件不随包分发。省略 `n` 时使用文件中的 `reference_n`，省略 `chromosomes` 和 `variant_scale` 时保留完整参考位点规模。全规模输出可能需要很大磁盘与生成时间；函数先检查估算工作集和可用空间，不能容纳时在创建输出目录前拒绝，不自动缩减为小数据。`variant_scale=0.001` 等显式比例可用于缩小测试，同时按比例缩小位点数和 mask 目标；它不再是完整参考规模。

首次生成前可只查看计划。`dry_run=True` 使用目标 n 和整数 AC / AN 构造少量人工帧，测量压缩后外推，**不创建输出目录**，也不要求磁盘已能容纳完整输出：

```python
from fudan_wgs_toolkit import generate_simulated_data

storage_plan_report = generate_simulated_data(
    output_directory="work/simulated_reference",
    calibration_profile="/private/calibration/simulation_profile.json",
    cpu_threads=2,
    memory_limit_gib=8,
    dry_run=True,
)
resource_plan = storage_plan_report["resource_plan"]
print(resource_plan["estimated_prepared_disk_bytes"])  # 压缩后最终文件的估计。
print(resource_plan["estimated_prepared_peak_bytes"])  # 含规划余量和临时文件的准入值。
print(resource_plan["generation_fits_available_disk"])
```

## 3. 参数与校准文件

### 全部参数

| 参数 | 默认值 | 意义与格式 |
| --- | --- | --- |
| `output_directory` | 必填 | 字符串或 `Path`；生成时必须是不存在的新目录，不覆盖已有输入或恢复半成品。dry_run 只定位目标文件系统，不写入。 |
| `n` | `None` | 整数且至少 128。None 使用校准文件的 `reference_n`；small 使用 1,000。 |
| `calibration_profile` | `None` | 私有校准 JSON 路径或 Python 字典。reference 模式可由环境变量或用户配置指定，small 不接受该参数。 |
| `preset` | `"reference"` | reference 使用私有汇总；small 为完全人工的 22 条染色体、每条 1,024 位点及 15 类 mask 测试预设。 |
| `chromosomes` | `"all"` | 校准文件中全部染色体，或不重复的 1–22 列表；接受整数、数字字符串及 `chr` 前缀。 |
| `variant_scale` | `1.0` | 正有限数。逐染色体位点数及 mask 目标按比例取整；1 保留校准规模。取整后的嵌套关系或 M<2 可以改变输出 mask 数，须查实际审计。 |
| `phenotype_type` | `"both"` | binary、continuous 或 both；两类表型分别写入 CSV。 |
| `n_binary_traits` | `2` | 选择二分类时的列数，整数且至少 2；未选择该类型时忽略。 |
| `n_continuous_traits` | `2` | 选择连续时的列数，整数且至少 2；未选择该类型时忽略。 |
| `seed` | `1729` | 非负整数；为染色体、位点、协变量和表型派生独立随机流。 |
| `phenotype_missing_rate` | `0.02` | 每个表型独立的缺失概率，范围 `[0,1)`；CSV 缺失为空单元格。 |
| `covariate_missing_rate` | `0.0` | 每个协变量单元格的缺失概率，范围 `[0,1)`；22 列共同完整的期望比例约 `(1-rate)^22`。 |
| `case_fraction` | `0.2` | 二分类 Logistic 概率的目标平均值，严格在 `(0,1)`；实际病例数由 Bernoulli 采样及缺失决定。 |
| `causal_effect` | `0.35` | 标准化人工 ALT dosage 的生成系数，有限实数；各表型交替使用正负号，0 表示该遗传项为零。 |
| `chunk_size` | `256` | 请求每个遗传帧的最大位点数，整数 1–1,024；根据主存预算可自动减小。 |
| `cpu_threads` | `8` | 染色体生成 worker 的上限，同时作为后续返回调用参数的 CPU 调度预算。 |
| `memory_limit_gib` | `8` | 生成阶段的主存估算预算，正有限数；1 GiB=`2**30` 字节。不代表 GPU 预算或操作系统 RSS 硬配额。 |
| `storage_limit_gib` | `None` | 可选正有限数，限定 prepared 的规划字节预算；None 使用峰值规划估计。CSV 空间另行预留；正式生成还检查可用磁盘及保留空间。低于峰值规划值时拒绝。 |
| `dry_run` | `False` | 布尔值。True 只返回资源计划，不生成 prepared / CSV；完整规模空间不足时仍可查看估计。主存和显式预算等参数仍须有效。 |

`cpu_threads` 不会重设已经初始化的 BLAS 线程池。需要限制 NumPy 原生线程时，在启动 Python 前设置，例如：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python run_simulated_example.py
```

内存估算覆盖样本向量、有限大小的 metadata 块、并发 worker 和紧凑遗传帧，不分配 `n × 整条染色体位点数` 的稠密矩阵。磁盘估计对每条染色体最多取12个稀疏度分层、每层最多8个人工位点，保持目标 n、AC 和 AN，以实际 row-delta / Zstd 编码测量并按频谱库存外推；metadata 按真实字段布局核算。人工独立调用与真实 LD 数据的压缩率可以不同。

最终大小估计、规划余量、临时峰值和未压缩上界分开报告。基因型规划上限采用探测估计的175%加固定余量，metadata 另加5%及固定余量；这些是准入余量，不是统计置信区间或完成保证。峰值还包括并发 metadata 字段从 `.raw` 复制为 NPY 时的临时共存。writer 逐帧检查实际字节预算及剩余磁盘；超出时保留未完成现场，不能把它作为完成数据读取。真实源数据准备的源 BED 10%基因型缓存规则保留，metadata 与原始输入另计。

新生成的六状态 CSR 帧对每个位点的有序样本索引做无损差分，再使用 Zstd level 3。它只改变存储表示，解码后样本索引、六状态、MAC、AN 和数值注释均保持原值。新版 reader 同时读取旧 absolute 帧 v1 和新 row-delta 帧 v2；新目录须配套新版安装。已有完成目录不就地改写，旧断点按原绑定格式恢复。prepared 根目录和 metadata 的 schema 不变，11个数值注释通道仍保留原来的 float64 表示。

### 私有文件解析顺序

reference 模式依次使用：

1. 显式 `calibration_profile` 路径或字典。
2. 环境变量 `FUDAN_WGS_SIMULATION_PROFILE` 指定的 JSON。
3. `FUDAN_WGS_SIMULATION_CONFIG` 指定的用户配置，未指定时为 `~/.config/fudan-wgs-toolkit/simulation.json`。

用户配置只需一项：

```json
{"calibration_profile": "/private/calibration/simulation_profile.json"}
```

配置中的相对路径相对于该配置文件；文件路径由使用者决定。校准文件使用 `format="synthetic-wgs-calibration-v1"`：

| 字段 | 结构与意义 |
| --- | --- |
| `reference_n` | 原始遗传样本轴大小，整数且至少128。用于 MAF / MAC 的样本量缩放。 |
| `chromosomes` | 染色体对象列表，`name` 为唯一的 1–22，`variants` 为该染色体物理位点数。 |
| `frequency_bins` | 每条染色体的联合频谱列表；每项含整数 `count`、`maf∈[0,0.5]`、`missing_rate∈[0,1]`。count 合计必须等于 variants。 |
| `minor_ac_mean`, `called_alleles_mean` | frequency bin 中可选的参考 MAC / 已调用等位基因数均值，供参考 n 下恢复整数计数。 |
| `minor_ac_low`, `minor_ac_high` | 可选的参考 MAC 分箱边界，只保存汇总，不包含个体调用。 |
| `masks` | 每条染色体的匿名目标列表，每项含 `kind`、非负整数 `group`、规范 `category` 和整数 `m≥2`；kind/group/category 组合唯一。 |
| `position_span` | 可选的染色体坐标跨度汇总，不小于 variants；只限定人工位置范围，不保留原始坐标。省略时人工跨度为 `2×variants`。 |
| `analysis_reference_n` | 可选的原分析样本数说明，与完整遗传 reference_n 区分；不改变遗传样本轴。 |

该格式只需要汇总统计。原始样本标识、疾病名、字段编号、坐标、基因名和关联 P 值均不作为生成输入。参考文件及生成后的逐位点 / 逐 mask 审计文件应留在私有存储。

## 4. 遗传、mask 与表型的生成方式

### MAF 保持与 MAC 缩放

每条染色体保留频谱各 bin 的位点库存；缩放位点数时采用总数守恒的分配。每个位点从剩余库存随机取一个 bin，根据 MAF 与缺失率确定已调用等位基因数 AN 和 MAC，再随机分配到人工样本的单倍型。参考样本量下可采用 bin 的参考计数均值；改变 n 时约为 `AN≈2n×(1−missing_rate)`、`MAC≈AN×MAF`，均取合法整数。

频谱 bin 的位点库存可以精确相同，bin 内使用均值不能恢复每个原始位点。低 MAC 位点随 n 变小可能变成 MAC=0，MAF 和稀有资格也受整数化影响；不把这类位点强行变成 MAC≥2。完整缺失及单等位基因缺失都使用成熟的六状态格式保存，持久化整数计数在每帧回读时核对。

### 15 类 mask

| 分析类型 | 规范 category |
| --- | --- |
| coding：7 类 | `plof`、`plof_ds`、`missense`、`disruptive_missense`、`synonymous`、`ptv`、`ptv_ds` |
| noncoding：7 类 | `upstream`、`downstream`、`UTR`、`promoter_CAGE`、`promoter_DHS`、`enhancer_CAGE`、`enhancer_DHS` |
| ncRNA：1 类 | `ncRNA` |

coding 使用人工 stopgain、splice、missense、synonymous 及 frameshift Indel 注释构造交集。disruptive missense 可以与 splice 重叠，所以 `plof_ds` 是集合并集，M 为两类 M 之和减去交集，`ptv_ds` 同理；不能只把两个 M 相加。官方没有发布的类别记为未知，不能据此断定 M=0 或 M<2。生成器从已知边际 M 推断一组可行的人工交集与未知类别，无法满足时报告不兼容目标。

各 noncoding mask 由人工独立的注释原子构造。已发布 M 只能约束边际数量，不能恢复原始变异交集、同一变异属于多个基因的关系或 LD。其余物理位点作为人工背景保留，仍可进入 Single 的正常筛选。

prepared 保存11个数值注释通道：`CADD`、`aPC.LocalDiversity`、`aPC.Epigenetics`，以及8个明确人工命名的 `Synthetic.Score01`–`Synthetic.Score08`。通道数量与参考计算配置一致，数值由人工分布生成，不匹配真实注释值或注释分布，也不复制额外数据库字段。人工类别与 QC 使用生产 selector 读取；数值注释是否参与加权沿用上面的 variant / SNV 规则。QC="PASS" 是合成标记，不代表执行真实遗传 QC。

`mask_counts_exact` 只核对**已发布目标行**在生成样本轴上的稀有资格 M，`mask_audit` 逐项保存 `source_m`、`target_m`、`observed_rare_m` 及 `matched`。当前参考清单含233,589条已发布目标，目标匹配不表示全部运行时 mask 行数相同。推断出的未知类别可能有 M≥2，正常关联会另外输出；`inferred_mask_audit` 和 `additional_inferred_eligible_mask_rows` 单独记录这些类别。

关联入口还会根据该列表型的有效样本、变异类型和 QC 重新筛选，所以生成审计不等于每个关联结果最终 M。改变 n、variant_scale 或缺失策略后，mask 数与 M 的偏离均应直接读取实际报告。

位置在给定跨度内分层随机采样，一层一个互异正整数；样本和基因采用人工前缀标签，等位基因为人工 SNV 或必要的 Indel。坐标不映射真实参考基因组。当前模型不重建真实 LD、亲缘、相位、重组史、跨 mask 交集和群体结构。

### 表型与协变量

所有染色体与 CSV 共用相同的有序 `FID,IID` 双列。协变量为 `age~Uniform(40,80)`、`sex∈{0,1}` 和独立标准正态的 `PC1`–`PC20`；这些 PC 是人工测试变量。由某个人工非恒定位点取得 ALT dosage，按均值和标准差标准化为 z：

\[
\eta=0.015(age-60)+0.15(sex-0.5)+0.08PC1+\beta z.
\]

连续表型为 `η+Normal(0,1)`；二分类求使平均 Logistic 概率接近 `case_fraction` 的截距，再独立 Bernoulli 采样。各列使用独立噪声与缺失位置。`ground_truth.csv` 保存生成系数与 dosage 尺度，可复核生成模型；它不提供预先知道的关联 P 值或有限样本估计系数。

## 5. 输出格式与返回值

```text
simulated_small/
  prepared/
    dataset.json / COMPLETE
    sample_pairs.npy / sample_ids.npy
    chr21/                      # 标准稀疏容器、计数、索引和 metadata。
    chr22/
    catalogs/                   # 人工 gene 任务与 promoter 区间。
  inputs/
    binary_phenotypes.csv
    continuous_phenotypes.csv
    covariates.csv
    ground_truth.csv
    causal_chr21.npy             # 各染色体找到的人工生成 dosage；不作为关联输入。
  simulation.json               # 全部生成及核对成功后发布。
```

| 输出 | 格式与意义 |
| --- | --- |
| `prepared/` | schema version 2 的生产 prepared 格式，含六状态压缩容器、metadata、catalog、样本身份、SHA 和完成标志。两类关联入口直接读取，具体格式见[数据准备](wgs_data_preparation.md)。 |
| `inputs/*_phenotypes.csv` | UTF-8 CSV，前两列 FID/IID，随后至少两列 `sim_binary_*` 或 `sim_continuous_*`；二分类有限值为0/1，连续为有限实数，缺失为空。未选择的类型不生成。 |
| `inputs/covariates.csv` | UTF-8 CSV：`FID,IID,age,sex,PC1,...,PC20`，行序与 prepared 相同。 |
| `inputs/ground_truth.csv` | 人工表型的生成系数、人工因果位点、dosage 均值 / 标准差及截距；关联入口不读取。 |
| `inputs/causal_chr*.npy` | 人工 dosage 数组，用于表型生成与溯源，关联入口不读取。 |
| `simulation.json` | 最终 JSON 报告，含配置、校准 SHA、计数审计、资源估算、耗时、输入文件 SHA 和两类调用参数。 |

生成期间保留 `simulation.pending.json`；全部成功后以最终报告替换。生成失败时保留现场且不发布成功报告。参数和资源准入失败发生在建目录前；写出过程中失败的目录不能作为完成输入。

| 重要返回字段 | 含义 |
| --- | --- |
| `completed`, `synthetic`, `schema_version` | 生成完成、人工数据标记及报告版本；不表示关联已经运行。 |
| `configuration`, `n_samples`, `seed` | 实际生成参数、样本数和随机种子。 |
| `calibration` | reference / small 来源类型、是否有参考校准及校准文件 SHA-256；calibrated 不表示所有 mask M 都精确匹配。 |
| `prepared_directory`, 各 `*_csv` | 完成目录与 CSV 路径；未选择的表型 CSV 为 None。 |
| `binary_kwargs`, `continuous_kwargs` | 对应公开入口可直接接收的参数字典；未选择的类型为 None。 |
| `chromosome_reports`, `mask_audit` | 各染色体位点数、频谱库存、MAC / AN、mask 目标与实际计数、时间和存储量；详细文件留私有。 |
| `mask_counts_exact`, `frequency_counts_exact` | 已发布目标行的稀有资格 M 与频谱库存是否核对一致；mask 标志不覆盖未知类别的总行数。 |
| `preparation.mask_counts_exact_scope` | 明确记录 mask 完全匹配仅针对已发布目标行。 |
| `chromosome_reports[].inferred_mask_audit`, `preparation.additional_inferred_eligible_mask_rows` | 未发布类别的人工推断 M、实际稀有 M，以及额外可能输出的合格 mask 行数。 |
| `trait_reports`, `complete_covariate_samples` | 各列表型可用 / 缺失 / 病例 / 对照数，以及22列协变量均完整的人数。 |
| `resource_plan`, `preparation` | 内存 / 磁盘准入估算、实际 worker / 帧大小与 direct prepared 子报告。 |
| `resource_plan.estimated_prepared_disk_bytes` | genotype、metadata、catalog 和样本轴的最终逻辑文件大小估计，不含表型 / 协变量 CSV。 |
| `resource_plan.estimated_prepared_peak_bytes` | prepared 规划余量加 metadata 临时共存的空间准入值，不是实测峰值。 |
| `resource_plan.estimated_total_disk_bytes`, `estimated_total_peak_bytes` | 在上述最终 / 峰值估计中分别加入输入 CSV 等空间。 |
| `resource_plan.generation_fits_available_disk` | 目标文件系统剩余空间扣除保留量后，是否可容纳峰值规划值；dry_run 时可为 False。 |
| `resource_plan.prepared_preflight` | 统一的构建计划，含压缩探测范围、敏感性余量和单独的 `uncompressed_upper_bound_bytes`；不把未压缩上界当作最终文件预测。 |
| `measured_storage`, `preparation.measured_prepared_storage` | 已生成文件的逻辑字节、分配块字节及硬链接去重值。前者统计输出树，排除 `simulation.json` 和 pending 自报告；后者统计 prepared。均不是磁盘高水位。 |
| `generation_seconds`, `total_seconds` | 完整生成调用的时间，不含关联；不再有原始 PLINK 生成与转换两个阶段。 |
| `files` | inputs 文件的相对路径、大小及 SHA；prepared 使用自身逐帧与清单校验。 |

`results_binary/` 和 `results_continuous/` 仅在调用关联入口后创建。运行完成仍须检查各自返回报告、worker 退出、块提交及最终 CSV。

## 6. 命令行与原软件对应

所有生成参数使用对应的 `--连字符` 选项。可用安装后的 `generate_simulated_data` 或 `python -m fudan_wgs_toolkit.simulate`；正式生成的报告同时保存到文件并打印到标准输出，`--dry-run` 只打印计划。

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# 显式的小规模人工预设；生成完成即可直接进入关联。
generate_simulated_data \
  --output-directory work/simulated_cli \
  --preset small --n 1000 --chromosomes 21 22 \
  --phenotype-type both --n-binary-traits 2 --n-continuous-traits 2 \
  --phenotype-missing-rate 0.02 --covariate-missing-rate 0 \
  --case-fraction 0.2 --causal-effect 0.35 \
  --seed 1729 --chunk-size 256 --cpu-threads 2 --memory-limit-gib 8

run-WGS-all-binanry \
  work/simulated_cli/inputs/binary_phenotypes.csv \
  work/simulated_cli/inputs/covariates.csv \
  work/simulated_cli/prepared \
  --output-directory work/simulated_cli/results_binary \
  --chromosomes 21 22 --device cuda:0 --cpu-threads 2 \
  --analyses individual coding noncoding ncrna \
  --gene-variant-type variant --no-impute-missing --grm none

run-WGS-continuous \
  work/simulated_cli/inputs/continuous_phenotypes.csv \
  work/simulated_cli/inputs/covariates.csv \
  work/simulated_cli/prepared \
  --output-directory work/simulated_cli/results_continuous \
  --chromosomes 21 22 --device cuda:0 --cpu-threads 2 \
  --analyses individual coding noncoding ncrna \
  --gene-variant-type variant --no-rm-nan --grm none

# 完整参考规模：省略 n 使用 reference_n，省略 chromosomes 使用全部22条。
# 先只查看存储计划；不创建目录。
generate_simulated_data \
  --output-directory work/simulated_reference \
  --calibration-profile /private/calibration/simulation_profile.json \
  --phenotype-type binary --n-binary-traits 16 \
  --cpu-threads 2 --memory-limit-gib 8 --dry-run

# 核对空间后去掉 --dry-run，才会执行同规模生成。
```

本生成器输出的是本包 prepared 格式，没有直接对应的官方模拟 prepared 命令。官方 STAAR 流程读取 GDS/aGDS，再调用 `Individual_Analysis_PheWAS`、`Gene_Centric_Coding_PheWAS`、`Gene_Centric_Noncoding_PheWAS` 和 `ncRNA_PheWAS`；相同输入的官方 R 调用见[二分类原软件调用](wgs_binary.md#原-staar-软件调用)。本功能不启动 R、PLINK 或官方程序。

同一合成输入可固定输入 SHA、样本处理、零模型、权重及 mask，比较优化前后的 `−log10(P)`；应同时核查全部有限值与任一路径显著的并集。校准到已发布 MAF / M 分布提供的是接近目标负载的人工输入，不能将跨队列官方结果当作这些人工数据的数值真值。

## 7. 验证与 benchmark

2026-10-16 的存储标准化修订在目标 GPU 计算节点完成 **307项CPU测试，全部通过，pytest计时57.89秒**。测试进程的 CPU affinity 为8，原生 BLAS 线程为2，CUDA设备不可见。覆盖六状态 / 整数计数、v1/v2 编码、损坏检测与旧断点恢复、空间不足保护、metadata、15类mask、prepared输入及两类公开入口契约。测试耗时不作为完整关联速度。[本轮匿名验证记录](../validation/storage_standard_20261016.anonymous.json)保存存储和实际GPU验证的范围。

### 真实文件大小与原13.2 TiB估计的更正

只读核对22份原始GDS及对应既有prepared，样本轴为345,967、物理位点合计1,023,836,503。以下是文件长度与文件系统分配块的分别测量；既有目录未重编码：

| 已有文件范围 | 逻辑字节 | 实际分配字节 |
| --- | ---: | ---: |
| 22份原始GDS | 1,372,012,487,282 | 1,372,012,568,576 |
| 既有prepared基因型容器 | 1,595,610,260,804 | 1,595,610,668,032 |
| 对应metadata | 277,586,941,100 | 277,591,289,856 |

此前报告的14,517,782,875,204字节、约13.2 TiB，来自**两倍宽松未压缩CSR上界和768 B/位点的metadata预留**，被错误标为输出大小预测。它不是实际生成结果，也不是GDS转prepared的实测膨胀率。metadata现在依据约234 B/位点的实际字段布局、样本轴及catalog核算；基因型改用压缩探测估计，未压缩上界单列。

新版本在完整参考n和位点库存上执行2,112个人工位点的只读压缩探测，耗时52.783秒，得到：

| 全规模计划项 | 字节 | 约TiB（2⁴⁰字节） |
| --- | ---: | ---: |
| prepared最终逻辑大小估计 | 1,472,623,306,503 | 1.339 |
| 含余量的最终规划值 | 2,406,054,002,370 | 2.188 |
| 含临时共存的峰值规划值 | 2,407,410,990,930 | 2.190 |
| 单列的未压缩上界 | 7,103,173,582,560 | 6.460 |

这些是新布局下的外推与准入值；合成独立位点不重建真实LD，不能把真实GDS的压缩率直接套用。此次没有创建全规模输出目录，**未生成全参考规模数据**，也没有把2.190 TiB规划值称为实测磁盘峰值。

另对真实prepared中的110帧、102,743个位点做有界只读重编码：相同Zstd level 3下，基因型payload从171,540,528降至125,723,312字节，减少26.71%，解码后的CSR数组逐项精确相同。该检查不修改已有缓存，也不外推成全量实测节省率。

### 目标节点上的有界生成与准备

以相同参考n=345,967和8,192个人工位点，对照旧absolute与新row-delta存储。已发布的15个mask目标全部匹配；所有遗传状态、整数计数、metadata、样本轴和表型CSV精确相同，没有写出原始PLINK：

| 直接生成 | 旧版 | 新版 |
| --- | ---: | ---: |
| 基因型payload字节 | 26,548,484 | 9,507,871 |
| prepared逻辑字节 | 197,424,842 | 180,385,828 |
| prepared实际分配字节 | 124,256,256 | 183,312,384 |
| 调用耗时 | 35.912秒 | 35.514秒 |

payload减少64.19%。该对照中，SHA相同且未改变内容的样本轴文件具有不同的物理分配量，因此**不能据此宣称直接生成的整个目录实际分配空间减少**。新版最终逻辑大小估计为181,270,753字节，实测180,385,828字节，本有界样例的误差约0.49%；峰值规划值为212,557,914字节，未测量完整磁盘高水位。

再将同一人工数据写成BED，实际运行普通 `prepare_WGS_data`，固定2CPU、chunk=64和11个注释通道。源输入逻辑大小为722,577,600字节，其中BED为708,542,467字节；两版基因型缓存的正常准入上限均为70,854,246字节，metadata另计：

| 普通prepare | 旧版 | 新版 |
| --- | ---: | ---: |
| 基因型payload字节 | 26,672,608 | 9,573,753 |
| prepared逻辑字节 | 199,428,271 | 180,488,807 |
| 去重后实际分配字节 | 206,970,880 | 183,836,672 |
| 调用耗时 | 63.622秒 | 63.400秒 |
| 进程峰值RSS字节 | 785,739,776 | 773,902,336 |

本样例实际分配空间减少11.18%。2,834,161,664个遗传状态、所有整数计数、196,608个metadata值及样本顺序精确相同，完成的压缩流均通过SHA核验。新版在根完成清单验证后移除 `.raw_annotations` 和 `.preflight` 暂存，并保留注释来源字节与SHA；旧版保留的raw annotation暂存为1,862,564逻辑字节、2,539,520分配字节。`storage.report.json` 对源输入、基因型缓存、metadata、暂存、硬链接去重和阶段快照分别计量，阶段快照不冒充真实峰值。

上述时间均来自共享节点上的一次有界对照，缓存冷热未严格控制；它们用于记录执行范围，不构成速度提升结论。

### 实际GPU入口回归

用户批准共享GPU上的小规模验证后，两版分别实际执行4种入口 / 缺失策略：binary available、binary impute、continuous available和continuous mean-imputed。二分类 `impute_missing=True` 将缺失标签填为对照；连续 `rm_nan=True` 在协变量匹配后，按各表型可用值的均值填补缺失标签，`rm_nan=False` 使用各表型的可用样本。每版共120份CSV、每种831行，合计3,324行；全部120份CSV解压后的字节逐一相同，15类gene类别均有有限P值。自有进程显存峰值小于60 MiB，关联显存准入预算为12 GiB；整卡共享占用不计入该进程的自有峰值。

该结果建立同一合成输入的旧 / 新存储兼容性和有限范围关联输出一致；本轮没有执行官方软件同输入P值对照，也没有运行十亿位点的完整GPU分析。

### 历史记录

[前一direct prepared修订记录](../validation/simulated_prepared_20261016.anonymous.json)对应267项CPU测试及原先GPU尚未获批的状态；其中13.2 TiB条目按上述原因更正，不再作为当前输出预测。该版本还记录了n缩小至4,096时低MAC取整为零、15个目标M随之改变的有界样例，未提高MAC制造相同M。

本轮保留已有metadata SHA缓存修复：共享文件系统mtime / ctime尚未稳定2秒时重新计算内容SHA，避免同大小、同时间刻度改写误命中；稳定的不可变文件仍可复用缓存。相关回归包含在本轮307项测试中。

[历史原始PLINK生成版](../validation/simulated_data_20261016.anonymous.json)固定于提交 `9ae80ba3eafc9d24a9635f332170f398bbb6a7a7`，不作为当前direct prepared版的通过记录或速度结果。

## 8. 更新记录

| 日期 | 更新与验证范围 |
| --- | --- |
| 2026-10-16，存储标准化 | 无损row-delta帧与追加日志；统一最终 / 峰值 / 未压缩上界及实际块大小统计，增加dry_run；普通prepare清理已提交暂存并保留来源。307项CPU测试、真实帧有界重编码、参考n的8,192位点生成 / BED准备及两版各4种GPU入口通过；120份结果CSV解压字节相同。全参考规模未生成。 |
| 2026-10-16，前一direct版 | 直接生成prepared；私有参考规模 / MAF与15类mask校准；11个人工数值注释通道，已发布 / 推断mask审计及SHA缓存修复。267项CPU测试与有界校准通过，当时GPU未运行；磁盘预留曾误标为输出估计，本版已更正。 |
| 2026-10-16，历史PLINK版 | 提交 `9ae80ba` 采用原始PLINK和独立人工注释再准备；149项回归、4次小规模GPU入口及10万人输入准备记录仅适用于该实现。 |

## 9. 参考与源码

- [生成器入口](https://github.com/weikanggong1/Fudan-WGS-Toolkit/blob/main/fudan_wgs_toolkit/simulate.py)、[私有汇总格式与预设](https://github.com/weikanggong1/Fudan-WGS-Toolkit/blob/main/fudan_wgs_toolkit/simulation_profile.py)、[直接 prepared 构建](https://github.com/weikanggong1/Fudan-WGS-Toolkit/blob/main/fudan_wgs_toolkit/simulate_prepared.py)。
- [prepared 格式与生产数据准备](wgs_data_preparation.md)、[二分类入口](wgs_binary.md)、[连续入口](wgs_continuous.md)。
- [STAARpipelinePheWAS 官方实现](https://github.com/li-lab-genetics/STAARpipelinePheWAS)：对应四类分析及 mask 定义。
- Li et al. Dynamic incorporation of multiple in silico functional annotations empowers rare variant association analysis of large whole-genome sequencing studies at scale. *Nature Genetics* (2020). [DOI:10.1038/s41588-020-0676-4](https://doi.org/10.1038/s41588-020-0676-4)。
- Li et al. A framework for detecting noncoding rare-variant associations of large-scale whole-genome sequencing studies. *Nature Methods* (2022). [DOI:10.1038/s41592-022-01640-x](https://doi.org/10.1038/s41592-022-01640-x)。
