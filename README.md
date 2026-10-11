# Fudan WGS Toolkit

> 本工具尚在开发验证阶段，未达到原软件精度，切不可作为真实分析使用的标准化工具。

Fudan WGS Toolkit 使用 PyTorch GPU 完成单变异、gene-based coding、noncoding 和 ncRNA 关联分析。`prepare_WGS_data` 从原始 PLINK 文件与独立功能注释准备遗传数据；`run_WGS_all` 读取表型 CSV、协变量 CSV 和完成的遗传目录，输出结果 CSV 分片。已有完成目录可直接分析，无需再次转存。

单表型和多表型使用同一个入口。多个表型共享遗传读取、缓存和设备传输，各自处理缺失样本、协变量和零模型；支持单卡或多卡、稀疏 GRM 或无亲缘结构模型，以及验证完成块后的恢复。运行和数据准备均使用 Python。

## 安装

```bash
git clone https://github.com/weikanggong1/Fudan-WGS-Toolkit.git
cd Fudan-WGS-Toolkit
conda env create -f environment.yml
conda activate fudan-wgs-toolkit
```

[完整使用说明、输入格式、流程图与验证记录](docs/fudan_wgs_toolkit.md)
