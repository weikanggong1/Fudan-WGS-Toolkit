# PairHMM source and scientific contracts

此目录自行实现三状态 PairHMM，没有复制发布 GATK/GKL 源码。官方固定版本：

|契约|固定源码|Git blob SHA|
|---|---|---|
|调用顺序与读段优先排列|[GATK VectorLoglessPairHMM.java](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/utils/pairhmm/VectorLoglessPairHMM.java)|9d6617800cf359c09dbb561867af08c7e205465a|
|固定依赖|[GATK build.gradle](https://github.com/broadinstitute/gatk/blob/4.2.0.0/build.gradle)|274334652d54f4455c41831039982779cd71b6ca|
|原生接口生命周期|[PairHMMNativeBinding.java](https://github.com/broadinstitute/gatk-native-bindings/blob/1.0.0/src/main/java/org/broadinstitute/gatk/nativebindings/pairhmm/PairHMMNativeBinding.java)|8925a3df4d65ed74cc6ffcb1855a1397a537032d|
|质量表与初始常数|[GKL Context.h](https://github.com/IntelLabs/GKL/blob/0.8.6/src/main/native/pairhmm/Context.h)|317ad1138936c89581b53c8f9f39e8946d3c97b5|
|递推、N、三态校正与求和|[GKL avx-pairhmm-template.h](https://github.com/IntelLabs/GKL/blob/0.8.6/src/main/native/pairhmm/avx-pairhmm-template.h)|a46fe9d5ed7e58f74854dbf1da1f54d94f3b852a|
|输出 log10 与双精度重算|[GKL IntelPairHmm.cc](https://github.com/IntelLabs/GKL/blob/0.8.6/src/main/native/pairhmm/IntelPairHmm.cc)|039713f29952a1203d2a613ed66c47d0687879b7|
|低概率阈值|[GKL pairhmm_common.h](https://github.com/IntelLabs/GKL/blob/0.8.6/src/main/native/pairhmm/pairhmm_common.h)|f996c3c3ef851d8c5db265461da29b1479ab6b6b|
|104对公开期望似然|[GKL pairhmm-testdata.txt](https://github.com/IntelLabs/GKL/blob/0.8.6/src/test/resources/pairhmm-testdata.txt)|747bbd33fd019df0c8c43eccb3a7d8cd0d1aab61|

GATK 4.2.0.0 依赖 GKL 0.8.6 与 gatk-native-bindings 1.0.0。
质量使用原始 Phred 值并按 `&127` 查询。N 视为匹配；不匹配排放概率除以3。
match-to-match 概率使用独立 float/double Jacobian 查表，步长1e-4、容差8。
FP32 初值 `2^120 / haplotype_length`，double 初值 `2^1020 / haplotype_length`。
最终对最后一行 M 与 insertion 状态分别求和，去掉初始常数的 log10。
缩放后的 FP32 概率低于 `1e-28f` 时，按 GKL 规则显式以 CPU double 重算并计数。
GPU 故障直接报错；请求全双精度 GPU 时直接拒绝。

组件自测包含人工解析例、变长序列、N、indel、Phred无符号掩码、低概率和批次余数。
这些是数值组件测试，不是端到端真实数据 benchmark。

参考：McKenna et al. Genome Research 20:1297–1303 (2010), doi:10.1101/gr.107524.110；
DePristo et al. Nature Genetics 43:491–498 (2011), doi:10.1038/ng.806。

This bundle includes only the fixed packed-H2D initialization-barrier implementation and the adaptive Java/JNI interface. The unsafe wavefront v1 source is excluded. The recurrence is FP32 (not TF32), with counted CPU-double recalculation only for the upstream low-probability rule. GKL remains an external Conda-installed CPU dependency.
