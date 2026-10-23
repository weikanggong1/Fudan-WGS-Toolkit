# 构建原版与直接追加版 fastp

`build_preprocessing_fastp` 使用已有 Conda 环境的 C++ 编译器、make 和 zlib，在新目录构建固定的官方 fastp 0.20.0。可选择原版 `original` 或源码优化版 `append`；后者减少普通双端通过记录格式化中的临时字符串和复制。两者使用相同编译选项和原过滤规则。构建不读取 FASTQ，也不更换环境中已安装的 fastp。

本接口自 v0.13.0 提供；v0.12.0 不包含这些源码资产参数。单样本完整输入、输出与运行示例见 [single_subj_preprocessing](single_subj_preprocessing.md#source-optimizations)。

```mermaid
flowchart LR
    A[固定官方源码 SHA] --> B[新目录内完整验证]
    B --> C[original 或 append 补丁]
    C --> D[已有 Conda C++ / zlib 编译]
    D --> E[C++ 字节和异常契约]
    E --> F[可执行文件 / 实际动态库 / 构建收据]
```

## Python 调用与输入输出

Conda 环境需已有 `gxx_linux-64`、`make` 和 `zlib`；此函数不安装或升级软件。只支持 Linux x86-64 ELF。构建及 Conda 路径限定为 ASCII 字母、数字和 `/ . _ + -`，防止 make/linker 将路径解释为指令或搜索路径分隔符。

```python
from fudan_wgs_toolkit.build_preprocessing_fastp import build_preprocessing_fastp

fastp_assets = build_preprocessing_fastp(
    conda_prefix="/opt/conda/envs/wgs",  # 已有环境，编译期间保持不变
    build_dir="/work/build/fastp_append",  # 必须是新目录
    variant="append",                  # original 为相同源码的原版对照
    cpu_affinity=[2],                   # 当前进程获准使用的 CPU 编号
    host_memory_limit_gib=2,            # 任务进程树 PSS 预算
    host_memory_reserve_gib=4,          # 共享节点保留主存
)
```

| 参数 | 含义、格式与默认值 |
| --- | --- |
| `conda_prefix` | 已有 Conda 环境目录；必须含 `conda-meta`、Conda C++ 编译器、make、zlib 头文件和共享库。 |
| `build_dir` | 新构建目录，不能在 Conda 环境或项目资源目录内，也不能覆盖既有目录。 |
| `variant` | `append`（默认）应用固定三文件补丁；`original` 编译固定官方源码。每个 variant 使用独立目录与收据。 |
| `source_archive` | 可选本地官方源码压缩包；默认下载。两条路径都校验同一固定字节数和 SHA-256。 |
| `cpu_affinity` | 1–4 个获准的 CPU 整数编号；默认取当前亲和性中的第一个。请求不能超过实际有限祖先 CPU 配额。 |
| `host_memory_limit_gib` | 1–16 GiB，默认 2；构建时检查进程树采样 PSS。 |
| `host_memory_reserve_gib` | 1–1024 GiB，默认 4；准入同时检查共享主存及至少 512 MiB 可用磁盘。预算和预留不是独占资源或独立 cgroup 硬配额。 |

返回字典 `fastp_assets`。新目录保留官方源码压缩包、完整源码、可执行文件、C++ 契约程序、逐阶段日志，以及 `build_receipt.private.json` 和最终提交的 `assets.private.json`。资产绑定原始源码、补丁、许可、工具、构建参数、实际动态库、ELF loader 及其文件身份和 SHA-256。失败时保留失败收据及现场，不提交可用 assets；重试必须使用新目录。不要公开含真实路径和身份的私有收据。

## 命令行调用

```bash
python -m fudan_wgs_toolkit.build_preprocessing_fastp \
  --conda-prefix /opt/conda/envs/wgs \
  --build-dir /work/build/fastp_original \
  --variant original --cpu-affinity 2 \
  --host-memory-limit-gib 2 --host-memory-reserve-gib 4
```

原软件调用保持普通双端 fastp 的选项：

```bash
fastp -i sample_R1.fastq.gz -I sample_R2.fastq.gz \
  -o clean_R1.fastq.gz -O clean_R2.fastq.gz \
  -j fastp.json -h fastp.html -w 1
```

默认 `-w 1` 保持确定输出顺序。不增加过滤、adapter、压缩或质量修剪参数。原 fastp 多 worker 的输出顺序本身可能变化，不能用多线程输出顺序差异替代过滤差异。

## 接入与 resume 验证

在单样本入口中传入 `fastp_source_patch_assets=fastp_assets`，或传入构建目录中 `assets.private.json` 的绝对路径；其余 FASTQ、参考和三份 known-sites 等参数照常提供。默认 `None` 使用环境中原有 fastp，完整资产会进入运行计划并在阶段使用及收尾时复验。

`fastp_argv` 返回上述普通 argv；`validate_fastp_assets` 仅核验。调用者须传入实际运行环境，例如 pipeline 的 `run.env`，并把完整 assets 纳入冻结配置。每次调用和 resume 跳过已完成阶段前都需核验。

```python
from fudan_wgs_toolkit.build_preprocessing_fastp import fastp_argv

# runtime_environment 是执行该命令时实际传给 subprocess 的完整环境字典。
command = fastp_argv(
    fastp_assets,
    read1="sample_R1.fastq.gz", read2="sample_R2.fastq.gz",
    output1="clean_R1.fastq.gz", output2="clean_R2.fastq.gz",
    json_report="fastp.json", html_report="fastp.html",
    runtime_env=runtime_environment,
    workers=1,
)
```

验证程序读取 ELF interpreter 并通过已绑定 loader 的 `--list` 只列出实际解析库，要求实际 libz 及全部其他库的路径、身份、SHA 与构建收据一致。即便 Runner 的 `LD_LIBRARY_PATH` 与编译时不同，也不能悄悄切换 zlib。`LD_PRELOAD`、`LD_AUDIT` 等动态装载覆盖项被拒绝。此核验是运行前检查；调用期间仍须保持冻结文件不被并发改写。

## 验证范围与更新

本构建器逐次运行原版 22 条或补丁版 35 条真实 C++ 序列化契约，包括四行字节、超长元数据、非空前缀、重复追加、长度溢出、分配失败后目标保持不变和合法重试。Python 契约覆盖压缩包路径/重复项拒绝、精确补丁、构建失败不发布资产、动态库变更及 resume 绑定。这些检查不读取生物数据，不是速度测试。

公共构建器的原/补两版均已实际编译并执行 C++ 契约；使用其产物完成一例真实 **1,000 万对前缀**的原–补–补–原 QC 对照。相同 4 CPU 预算、`-w 1` 和 gzip 4，CLI 中位数为 **145.588→140.229 s**（本组减少 3.68%）；四次完整 gzip 与有序解码字节、配对 ID 和科学 JSON 一致，JSON 只排除 `command` 路径字段。采样任务 PSS 最大 **1.120 GiB**。独立完整回读 **184.303–186.066 s/次**，在 CLI 计时外；两版各两次、共享节点且未清空文件缓存，不称稳定或完整流程加速。详细计时边界见[主文档 9.4](single_subj_preprocessing.md#94-小组件结果与未采用方案)。

新构建成功不自动继承其他可执行文件的精度或速度证据，完整单样本、VCF、gVCF、PLINK 和长期内存结果仍需相应端到端验证。返回字段 `scientific_validation_completed=False` 表示构建本身不执行科学验收。

本次新增固定 0.20.0 原版/补丁版独立构建、完整源码与 MIT 许可绑定、实际 libz 环境验证及失败收据；保留默认单 worker 和原过滤规则。

## 参考

- [fastp v0.20.0 官方源码](https://github.com/OpenGene/fastp/tree/v0.20.0)，[原始 MIT 许可](https://github.com/OpenGene/fastp/blob/v0.20.0/LICENSE)。
- Chen S, et al. *fastp: an ultra-fast all-in-one FASTQ preprocessor*. Bioinformatics (2018). [doi:10.1093/bioinformatics/bty560](https://doi.org/10.1093/bioinformatics/bty560)。
