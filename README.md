# Fudan WGS Toolkit

Fudan WGS Toolkit 使用 PyTorch GPU 进行多个二值表型的全基因组关联分析，支持 Single、coding、noncoding 和 ncRNA。`prepare_WGS_data` 从原始 PLINK 文件与独立功能注释准备遗传数据；`run_WGS_all_binanry` 复用完成的遗传目录，输出每个表型独立的 CSV 结果。

输入表型仅接受 `0`、`1` 和缺失值。默认把缺失表型填为 `0`（对照），也可按每个表型的实际有值样本分析。默认共享同一套协变量；多 GPU 按整条染色体分工，共享遗传读取与有界缓存，并支持完成块校验后的恢复。

## 安装

```bash
git clone https://github.com/weikanggong1/Fudan-WGS-Toolkit.git
cd Fudan-WGS-Toolkit
conda env create -f environment.yml
conda activate fudan-wgs-toolkit
```

[使用说明：输入格式、全部参数、流程图、示例与验证记录](docs/fudan_wgs_toolkit.md)
