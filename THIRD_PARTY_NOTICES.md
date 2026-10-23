# 原软件源码优化与第三方许可

Fudan WGS Toolkit 的项目代码使用 [GPL-3.0-only](LICENSE.md)。v0.13.0 预处理实现在固定原软件版本上提供小型源码补丁、构建器和契约；原软件及其依赖各自的许可继续适用。

| 组件 | 固定来源 | 许可与仓库内保留文件 |
| --- | --- | --- |
| fastp 0.20.0 | [官方 v0.20.0](https://github.com/OpenGene/fastp/tree/v0.20.0) | MIT；[原许可](fudan_wgs_toolkit/_preprocessing_fastp/FASTP_LICENSE)，[源码与补丁清单](fudan_wgs_toolkit/_preprocessing_fastp/SOURCES.md)。 |
| BWA 0.7.17 | [官方 commit 9f26bfcc](https://github.com/lh3/bwa/tree/9f26bfcc7780753129b60717ecab0ebba6f04b7c) | GNU GPLv3；[原 COPYING](fudan_wgs_toolkit/_preprocessing_bwa/BWA_COPYING)，[源码与补丁清单](fudan_wgs_toolkit/_preprocessing_bwa/SOURCES.md)。官方树中 `ksw.c` 的 MIT 声明在构建源码中保留。 |
| GATK 4.2.0.0 的两个 BQSR 训练类 | [官方 commit 1f7fade2](https://github.com/broadinstitute/gatk/tree/1f7fade2afbd706539eb983087907a0cb11ee734) | Apache-2.0；[原许可](fudan_wgs_toolkit/_preprocessing_gatk_sourcepatch/LICENSE.GATK-4.2.0.0.txt)，[源码、修改范围和类加载说明](fudan_wgs_toolkit/_preprocessing_gatk_sourcepatch/SOURCES.md)。 |

每个包的 `source_manifest.json` 绑定官方源码、最小补丁、许可及契约的 SHA-256。公开包仅含这些补丁、清单、许可和项目构建/验证代码；不包含完整原软件源码、可执行文件、原 GATK jar、Conda 环境、参考基因组、known-sites 或参与者测序与遗传结果。构建器从固定官方来源获取原软件，严格核对后在用户的新目录中编译，保留原始源码、许可和修改记录。

fastp/BWA 的原版与补丁版独立构建使用同一环境和编译参数；GATK 产生独立 overlay jar，原安装 jar 保持不变。生成物的使用、修改或再分发仍须遵守相应原许可。构建环境的 zlib、编译器、Java 和原 GATK jar 内各依赖的许可不由本项目替代；本项目不重新打包这些运行库。

可选 CUDA/JNI PairHMM 的来源及其 GATK/GKL/native-bindings 许可另见[该组件说明](fudan_wgs_toolkit/_preprocessing_gpu/THIRD_PARTY_NOTICES.md)。GATK 4.2.0.0 本体为 Apache-2.0，不能与其他 native 接口的 BSD 许可混同。
