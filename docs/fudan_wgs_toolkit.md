# Fudan WGS Toolkit 文档索引

功能说明分别维护在以下文档中。每份文档包含输入与输出格式、全部参数、Python 与命令行示例、流程图、原软件对应调用、真实验证与 benchmark、参考文献。

| 功能 | 入口 | 说明 |
| --- | --- | --- |
| 单样本预处理 | `single_subj_preprocessing` | [配对 FASTQ、参考资源、GVCF/VCF 与逐染色体 PLINK](single_subj_preprocessing.md) |
| 数据准备 | `prepare_WGS_data` | [原始 PLINK、独立注释与完成遗传目录](wgs_data_preparation.md) |
| 二分类 WGS | `run_WGS_all_binanry` | [严格 0/1 输入、缺失标签处理、Burden / SPA 与恢复](wgs_binary.md) |
| 连续表型 WGS | `run_WGS_continuous` | [均值填补或 available 样本、连续零模型、完整谱与恢复](wgs_continuous.md) |

已有完成遗传目录可直接进入关联分析。缓存身份格式、GRM 绑定和结果恢复分别写在对应功能文档中。

安装见[项目主页](../README.md#安装)，验证指标的含义见[验证口径](../validation/README.md)。
