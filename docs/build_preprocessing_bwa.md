# 构建原 BWA 与 mate-profile 复用版

`build_preprocessing_bwa` 从固定官方 BWA 0.7.17 commit 构建 `original` 或 `mate_profile`。后者在同一 mate 的多个救援候选之间复用 query profile，减少重复准备及临时分配；原动态规划、tie/rescue 顺序和配对统计保持不变。构建器不建立参考索引、不读取 FASTQ、不执行比对，也不改写 Conda 环境。

本接口自 v0.13.0 提供；v0.12.0 不包含这些源码资产参数。完整单样本输入、输出与 Python/CLI 示例见 [single_subj_preprocessing](single_subj_preprocessing.md)。

```mermaid
flowchart LR
    A[固定官方源码与 GPL 许可] --> B[original 或 mate_profile 精确补丁]
    B --> C[同一 Conda GCC / 原 O2 编译]
    C --> D[实际原版与候选 C ABI 契约]
    D --> E[程序 / 动态库 / 独立构建收据]
```

```python
from fudan_wgs_toolkit.build_preprocessing_bwa import build_preprocessing_bwa

bwa_assets = build_preprocessing_bwa(
    conda_prefix="/opt/conda/envs/wgs",  # 已有的 GCC/ar/make/zlib 环境
    build_dir="/work/build/bwa_mate_profile",  # 新目录，不覆盖旧构建
    variant="mate_profile",  # original 使用固定官方源码
    cpu_affinity=[2, 3],
    host_memory_limit_gib=2,
    host_memory_reserve_gib=4,
)
```

| 参数 | 含义与格式 |
| --- | --- |
| `conda_prefix` | 已有 Conda 环境目录；需含 Conda GCC、ar、make，以及 zlib 头文件和共享库。函数不安装依赖。 |
| `build_dir` | 新目录，位于 Conda 和打包资源目录之外；失败保留现场，新目录重试。 |
| `variant` | `mate_profile`（默认）或 `original`。每次只提交一个可执行文件的独立资产与收据。 |
| `source_archive` | 可选本地官方源码包；省略则从官方固定 commit 下载，两者核验相同字节数和 SHA。 |
| `cpu_affinity` | 1–4 个获准的 CPU 整数编号；默认取当前亲和性第一个，不超过任何实际有限 CPU 配额。 |
| `host_memory_limit_gib` | 构建进程树 PSS 预算，1–16 GiB，默认 2。 |
| `host_memory_reserve_gib` | 共享节点主存预留，1–1024 GiB，默认 4；同时检查至少 512 MiB 可用磁盘。这是共享准入预算，不是独占或独立 cgroup 硬配额。 |

只支持 Linux x86-64。构建和 Conda 路径限定为 ASCII 字母、数字及 `/ . _ + -`。返回 `bwa_assets` 字典，含明确的 `executable` 文件身份/SHA 绑定、实际 ELF loader 与动态库、原版/候选完整源码闭包、GPL 许可、构建/ABI 契约版本及实际退出收据。新目录保留源码、独立原版 oracle 库、契约程序及日志。原始 Makefile 不变；两版均使用原 `-O2`，并同样加 `-fcommon` 以兼容现代 GCC 的旧版全局变量规则。

ABI 契约需要将实际候选 C 文件与原版 `libbwa.a` 比较，因此候选 build 会另构建原版 oracle 库；原版 build 则创建候选契约源码副本。这些步骤均计入构建预算，不产生额外真实数据分析。原 BWA 不带参数时打印 `0.7.17-r1188` 并返回 1，构建器记录且严格核验该预期行为，不能把它计为成功比对。

命令行：

```bash
python -m fudan_wgs_toolkit.build_preprocessing_bwa \
  --conda-prefix /opt/conda/envs/wgs --build-dir /work/build/bwa_original \
  --variant original --cpu-affinity 2,3
```

接入前使用实际执行环境验证资产，并把完整资产纳入调用者的冻结配置；每次 resume 跳过已完成阶段前同样需要验证。

```python
from fudan_wgs_toolkit.build_preprocessing_bwa import validate_bwa_assets

validate_bwa_assets(bwa_assets, runtime_env=runtime_environment)
# 单样本入口使用以下参数；其余必填输入见主文档完整示例。
pipeline_options = {
    "aligner": "bwa",
    "bwa_source_patch_assets": bwa_assets,
}
```

`bwa_source_patch_assets` 也接受 `assets.private.json` 的绝对路径。入口从完整资产取得程序，不要求环境中另有 `bin/bwa`；参考 BWA 索引仍须提供。无需再填 `aligner_executable`，显式给出不同程序会拒绝。单独使用程序路径的调用者可取 `bwa_assets["executable"]["path"]`，仍须自行验证完整资产和实际运行环境。

验证复用 fastp 的 ELF/动态依赖核验实现并绑定其源码 SHA；若实际运行环境解析出的 libz 或其他库变化，则拒绝使用旧资产，需恢复构建环境或重新构建。更新共享 helper 后也需要新构建，不可只改标签继续 resume。

对应原软件命令仍为普通 BWA-MEM，例如：

```bash
bwa mem -t 4 -K 10000000 -R '@RG\tID:sample\tSM:sample' \
  reference.fa clean_R1.fastq.gz clean_R2.fastq.gz > aligned.sam
```

`-K`、线程数、读组、输入、索引及其他原选项必须在同配置对照中一致。此 builder 不添加这些分析参数，不拆分 batch 或改变配对推断。

两版公共构建器已实际编译并完成 C API 契约及预期 version exit1 核验。同一科学源码补丁此前用真实 1M 前缀完成原–补–补–原组件对照，每次 1,930,900 条有序 SAM 记录的全部字段一致，header 只归一化 `@PG CL` 的可执行文件路径。原版两次为 112.545、155.383 s，补丁为 95.254、99.140 s；共享负载下原版明显波动，不能据此称稳定加速。采样 PSS 最大均约 5.38 GiB，严格 SAM 回读另计。细节见[主文档 9.4](single_subj_preprocessing.md#94-小组件结果与未采用方案)。

此前科学源码的组件证据不自动替代新 build 的真实比对验证。完整 BAM/gVCF/VCF/PLINK 和长期内存仍需端到端核验，返回字段 `mapping_validation_completed=False` 表示构建本身不执行比对验收。

参考：[官方固定源码](https://github.com/lh3/bwa/tree/9f26bfcc7780753129b60717ecab0ebba6f04b7c)、[COPYING](https://github.com/lh3/bwa/blob/9f26bfcc7780753129b60717ecab0ebba6f04b7c/COPYING)、Li H. *Aligning sequence reads, clone sequences and assembly contigs with BWA-MEM*. [arXiv:1303.3997](https://arxiv.org/abs/1303.3997)。
