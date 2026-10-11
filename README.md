# Fudan WGS Toolkit

> 本工具尚在开发验证阶段，未达到原软件精度，切不可作为真实分析使用的标准化工具。

Fudan WGS Toolkit 使用 PyTorch GPU 完成全基因组单变异、gene-based coding、noncoding 和 ncRNA 关联分析。多个表型共享遗传数据读取、缓存和传输，每个表型使用自身的完整样本集合。`prepare_WGS_data` 从原始 PLINK BED/BIM/FAM 与独立功能注释准备数据；`run_WGS_all` 消费表型 CSV、协变量 CSV 和已完成遗传目录，输出四类最终 CSV。运行和数据准备均使用 Python。

正常流程是先运行 `prepare_WGS_data`，再运行 `run_WGS_all`。遗传数据只需准备一次；已有完成目录时，直接把表型 CSV、协变量 CSV 和该目录交给 `run_WGS_all`，省略准备阶段。所有表型可在一次调用中提交；混合 PheWAS 的 `trait_batch_size` 控制计算批次，不限制总表型数。

遗传目录的清单决定样本格式和模型：`dataset.json` 使用完整 FID/IID 和普通零模型；`cache_dataset.json` 使用原数字身份轴、缓存绑定的 GRM 和协变量 profile，执行混合模型 PheWAS。两种格式的输入及输出结构见主要使用说明。

## 安装

```bash
git clone https://github.com/weikanggong1/Fudan-WGS-Toolkit.git
cd Fudan-WGS-Toolkit
conda env create -f environment.yml
conda activate fudan-wgs-toolkit
```

## 已有遗传缓存时直接运行

下面的目录已包含完成的 `cache_dataset.json` 及其 GRM、样本轴和注释，无需再次运行 prepare：

```bash
run-WGS-all /data/phenotypes.csv /data/covariates.csv /data/completed_cache \
  --output-directory /data/results \
  --gpu-ids 0 1 --trait-batch-size 16 --continuous-transform paper
```

混合 PheWAS 按实际可用 GPU 内存准入，不设固定 20 GiB 上限；Burden 和二分类模型工作区使用统一预算与安全余量。各表型保留自己的有效样本和检验结果。已有模型的复用另需完整来源绑定，遗传缓存复用本身不会自动跳过模型拟合。

[完整使用说明、输入格式、流程图与验证记录](docs/fudan_wgs_toolkit.md)

[已完成缓存与混合模型 PheWAS 调用](docs/phewas_cache_compatibility.md) · [已拟合模型的来源与复用](docs/phewas_model_reuse.md)
