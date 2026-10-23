# fastp 0.20.0 源码与许可

`build_preprocessing_fastp` 从 [OpenGene/fastp 官方 v0.20.0 tag](https://github.com/OpenGene/fastp/tree/v0.20.0) 获取源码，不在包中分发原软件可执行文件或整份源码。源码压缩包为 139,422 字节，SHA-256 为 `8d751d2746db11ff233032fc49e3bcc8b53758dd4596fdcf4b4099a4d702ac22`。`source_manifest.json` 记录原始 73 个文件及三个修改文件的 SHA-256。

fastp 使用 MIT 许可；原许可完整保存在 `FASTP_LICENSE`，构建目录中保留官方 `LICENSE`。直接追加补丁 `fastp-0.20.0-append-direct.patch` 的 SHA-256 为 `0e4d8f6a523db4f656b1472070887ad54c2f280c425452b4d573b12f77f39442`。补丁仅在 `Read` 新增完整长度检查后的直接追加方法，并替换普通双端通过记录写入两个文件缓冲区的调用；原过滤、adapter、overlap、质量修剪及压缩实现保持不变。此版本原来通过 `Read::toString()` 产生临时字符串；不能把后续版本显式分配 `new[]` 的实现当作 0.20.0 的原代码。

`append_contract.cpp` 是项目的序列化/分配失败契约，分别对原版及补丁版实际 C++ 源码编译。它使用生成的边界字节输入验证输出及异常行为，不是生物数据或性能 benchmark。

原实现：[read.cpp](https://github.com/OpenGene/fastp/blob/v0.20.0/src/read.cpp)、[peprocessor.cpp](https://github.com/OpenGene/fastp/blob/v0.20.0/src/peprocessor.cpp)、[Makefile](https://github.com/OpenGene/fastp/blob/v0.20.0/Makefile)。
