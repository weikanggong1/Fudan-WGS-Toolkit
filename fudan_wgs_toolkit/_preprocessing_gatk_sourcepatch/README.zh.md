# 原 GATK BQSR 源码补丁

此可选构建修改原 GATK 4.2.0.0 的两个类，减少每条读段重复创建的对象。
输入为已有预处理 Conda 环境，环境须提供 GATK 4.2.0.0 和 JDK 8；输出为
环境外新建目录中的 overlay jar、源码清单、实际类加载凭据和已提交 assets。
原 Conda jar 保持原字节。构建不读取测序数据，不运行 BQSR 科学分析。

```python
from fudan_wgs_toolkit.build_preprocessing_gatk_sourcepatch import build_preprocessing_gatk_sourcepatch
from fudan_wgs_toolkit import single_subj_preprocessing

source_patch_assets = build_preprocessing_gatk_sourcepatch(
    conda_prefix="/opt/conda/envs/wgs-preprocess",  # 已配置的原软件环境。
    build_dir="/work/build/gatk-bqsr-overlay",     # 不存在的新目录；不能放在环境内。
    cpu_affinity=[0],                            # 实际获准使用的 CPU 编号，1–4 核。
    host_memory_limit_gib=4,                     # 构建进程树共享 PSS 预算。
    host_memory_reserve_gib=4,                   # 实际节点/有限祖先配额须另有此预留。
    upstream_source_dir=None,                    # None 下载；离线可给固定原源码缓存根。
)
# 将 source_patch_assets 传给常规函数的 gatk_source_patch_assets 参数。
# 其余 FASTQ、参考、三份 known-sites 等输入按主文档提供。
```

缓存根须包含 `src/main/java/.../BaseRecalibrationEngine.java`、
`src/main/java/.../ReadRecalibrationInfo.java` 与原 `LICENSE.TXT`；每个文件均须
命中固定 SHA，不接受替代版本。构建器按现有资源准入、线程亲和性、进程树
PSS 与退出码记录阶段；构建失败保留目录和诊断，不发布成功 assets。

```bash
python -m fudan_wgs_toolkit.build_preprocessing_gatk_sourcepatch \
  --conda-prefix /opt/conda/envs/wgs-preprocess \
  --build-dir /work/build/gatk-bqsr-overlay --cpu-affinity 0
```

`single_subj_preprocessing(..., gatk_source_patch_assets=source_patch_assets)`
也接受已提交 `assets.private.json` 的路径。默认 `None` 使用已有原/并行入口。
`bqsr_mode="original"` 在原 BaseRecalibrator JVM 中加载补丁；`"parallel"`
及选择并行的 `"auto"` 在现有 adapter JVM 中加载同一补丁，每个 worker 仍
独立持有原引擎。ApplyBQSR、HaplotypeCaller 和 QC 不加载此 overlay。

阶段输出增加 `bqsr_sourcepatch.private.json`，包含真实 CodeSource、class SHA
及入口；续跑同时核对源码、环境记录、jar、构建收据、该输出和原阶段命令。
修改其中任一绑定后需新建运行目录。构建目录和凭据使用绝对 Linux 路径，
不能移动构建产物后继续声称旧收据有效。实际数据与完整运行命令只保存在
私有运行目录；不要提交到公开仓库。

参考原命令为 `gatk BaseRecalibrator -R reference.fa -I reads.cram -O recal.table
--known-sites known1.vcf.gz ...`。补丁保持科学选项和事件轴，改变训练类的
加载方式。当前公开构建成功仅说明编译与类加载通过；真实数据精度、完整
pipeline 速度和长时间内存稳定性须分别验证。原实现及许可见 `SOURCES.md`。
