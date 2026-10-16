# Fudan WGS Toolkit

Fudan WGS Toolkit 使用 PyTorch GPU 完成多个二分类或连续表型的全基因组关联分析，支持 Single、coding、noncoding 和 ncRNA。遗传数据准备一次，两类分析直接复用完成的缓存；每个表型保存独立的 CSV 结果。

## 功能与说明文档

| 功能 | Python 入口 | 输入与默认行为 | 完整说明 |
| --- | --- | --- | --- |
| 数据准备 | `prepare_WGS_data` | 原始 PLINK BED/BIM/FAM 与独立功能注释，生成有校验和的遗传目录 | [数据准备](docs/wgs_data_preparation.md) |
| 二分类 WGS | `run_WGS_all_binanry` | 至少两个 0/1/缺失表型；默认缺失标签填为 0，可选各列 available 样本 | [二分类分析](docs/wgs_binary.md) |
| 连续表型 WGS | `run_WGS_continuous` | 至少两个连续数值表型；默认逐列均值填补，可选各列 available 样本 | [连续表型分析](docs/wgs_continuous.md) |
| 合成测试数据 | `generate_simulated_data` | 生成虚构表型、协变量、PLINK 与独立注释，默认完成数据准备，可直接进入两类关联入口 | [合成数据生成器](docs/generate_simulated_data.md) |

各功能文档介绍流程图、输入格式、全部参数、Python 和 CLI 调用、原软件对应调用、验证范围及参考文献。`run_WGS_all_binanry` 是当前公开函数名，调用时保留这一拼写。

```mermaid
flowchart LR
    RAW["原始 PLINK + 独立注释"] --> PREP["prepare_WGS_data"]
    PREP --> CACHE["完成的遗传目录"]
    CACHE --> BINARY["run_WGS_all_binanry"]
    CACHE --> CONTINUOUS["run_WGS_continuous"]
    PH["表型 CSV + 协变量 CSV"] --> BINARY
    PH --> CONTINUOUS
    BINARY --> CSV["每个表型独立的四类 CSV"]
    CONTINUOUS --> CSV
```

已有完整遗传目录时直接进入关联分析。两类入口默认共用协变量；二分类可逐表型指定协变量，连续入口使用全部协变量。协变量缺失均排除，不进行填补。各表型的零模型和检验矩阵独立，多 GPU 按完整染色体分工，使用有界缓存与节点共享主存预算。`resume=True` 校验已提交块后补算剩余分析。

## 安装

```bash
git clone https://github.com/weikanggong1/Fudan-WGS-Toolkit.git
cd Fudan-WGS-Toolkit
conda env create -f environment.yml
conda activate fudan-wgs-toolkit
```

Conda 环境包含 CUDA 11.8、PyTorch 2.5.1 和本包的 editable 安装。数据准备在 CPU 执行，关联矩阵计算优先使用原生 TF32/FP32；零模型和稳定概率计算保留各自所需精度。

## 如何开始

1. 尚无遗传目录：按[数据准备说明](docs/wgs_data_preparation.md)准备原始 PLINK 和独立注释。
2. 已有完成目录：准备带身份列的表型、协变量 CSV，选择[二分类](docs/wgs_binary.md)或[连续表型](docs/wgs_continuous.md)入口。
3. 按文档核对 `completed`、worker 退出、已提交块和最终 CSV；恢复时使用同一输出目录和输入配置。

两种表型分析的标签处理和模型不同，应分别调用相应入口。实测速度、精度及覆盖范围分别记录在各功能文档的“验证与 benchmark”中；[验证口径](validation/README.md)说明如何解读这些指标。
