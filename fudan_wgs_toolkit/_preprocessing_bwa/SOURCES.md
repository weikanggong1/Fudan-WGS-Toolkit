# BWA 固定源码与 mate-profile 补丁

官方源码：[lh3/bwa v0.7.17](https://github.com/lh3/bwa/tree/9f26bfcc7780753129b60717ecab0ebba6f04b7c)，commit `9f26bfcc7780753129b60717ecab0ebba6f04b7c`。从官方 codeload 下载的压缩包为 232,802 字节，SHA-256 `79763dd6c48e8b01b8e27d99e6e5575d8bbca419e3eed9f3e330af072e63b370`。`source_manifest.json` 逐文件绑定全部 77 个源文件，以及仅修改 `bwamem_pair.c` 后的闭包。

BWA 使用 GNU GPLv3；完整官方 `COPYING` 保存在 `BWA_COPYING`，其 SHA-256 为 `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903`。构建时保留完整官方源码及其中 `ksw.c` 的 MIT 许可声明。本包不分发整份原软件或可执行文件。

`mate_profile.patch` 的 SHA-256 为 `923b124eca08c9d5dd469f33eada27ee7c94ebcbede02704aeba27161e98ce48`。补丁在同一 mate 的救援期间复用两个方向的 query profile 和反向查询缓冲；工作区仅属于这一 mate，结束时释放，不引入全局或 TLS 缓存。原 DP、rescue/tie 顺序、配对统计、batch 大小和导出 `mem_matesw` API 均保留。

`mate_profile_contract.c` 对实际候选 translation unit 和原版 `libbwa.a` 比较，包含 3,360 组生成的 anchor 契约、四线程、跳过路径、query 恢复和所有者释放。生成输入用于 API/边界验证，不是生物学或性能 benchmark。`version_probe.py` 检查原 BWA 无参数调用的预期退出码 1，不将它误报为比对成功，也不接受任意非零退出。
