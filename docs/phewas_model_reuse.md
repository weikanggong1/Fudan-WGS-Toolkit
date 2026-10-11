# 已拟合混合模型的来源与复用

模型复用保留原模型的 metadata、数组及拟合源码 SHA。当前运行另记录消费这些模型的源码 SHA。数学兼容证明与实际文件导入证明分别验证，任一证明缺失或不匹配都会中止；复用模式不进入新的拟合路径。

计划 JSON 使用以下字段；既有命令行 `python -m fudan_wgs_toolkit.phewas_run --plan /path/to/plan.private.json` 读取该计划，不新增必须填写的命令行参数。

从表型 CSV、协变量 CSV 和完成遗传缓存开始的新运行，使用 [三个输入的调用示例](phewas_cache_compatibility.md)，默认拟合本次模型。本文的复用模式是已准备部署计划的来源协议，需要另行生成并审核兼容证明和完整导入清单；公开运行入口验证这些证明，不自动生成或伪造证明。每个计划 worker 可按明确分配执行，例如：

```bash
python -m fudan_wgs_toolkit.phewas_run \
  --plan /path/to/plan.private.json \
  --worker-id 0 --worker-count 1 --device cuda:0
```

多 worker 部署时，每个 worker 负责自身分配并等待全部模型覆盖，worker 数量与设备由部署方确定。完整关联成功还需要全部 worker 输出及 CSV 回读通过；已有模型验证完成不能代替关联完成。

| 计划字段 | 含义 |
| --- | --- |
| `reuse_models_only` | 明确为 `true`，仅验证并读取已有模型。缺失或错误的模型链接会硬失败。 |
| `source_implementation_sha256` | 当前消费源码的包级 SHA-256，必须与实际运行源码一致。 |
| `model_fit_source_implementation_sha256` | 原模型拟合源码的包级 SHA-256。原模型 metadata 的 `fit.source_implementation_sha256` 保持此值。 |
| `null_fit_compatibility_receipt` / `null_fit_compatibility_sha256` | 独立数学兼容证明的绝对路径及文件 SHA-256。证明绑定上述两套源码，并逐组件核验输入筛选、协变量设计、GRM、连续变换、Gaussian 与二分类混合模型代码的 fingerprint。 |
| `model_import_receipt` / `model_import_sha256` | 私有文件导入清单的绝对路径及文件 SHA-256。完整计划在生成该清单之后绑定它，避免循环哈希依赖。 |

导入清单为 `schema_version: 1`、`status: "verified"`，含两套源码 SHA、数学证明 SHA、`prepared_inputs_manifest_sha256`、`kinship_sha256`、`cohort_sha256`、`continuous_transform` 及 `model_count`。它同时记录原任务 `from_run_directory`、原模型目录 `from_models_directory`、新运行模型目录 `models_directory`、原计划 `from_plan_sha256`、`source_fit_worker_count` 和每个原 worker 的 `fit_supervisor_exit_sha256`、`fit_complete_receipt_sha256`。消费端会实际回读这些原退出和完成记录，核对退出码、源码、原计划及完整表型分配。

`models` 按 `trait_index` 从零连续排列。每项含 `target`（原规范目录）、`symlink`（当前运行的模型链接）、`family`、`n`、`metadata_sha256` 以及 `files`。每个文件项含绝对 `path`、`sha256` 和 `stat`；`stat` 包含 `ino`、`size`、`mtime_ns`、`ctime_ns`。不同节点挂载的 device number 可能不同，因此不把它当作跨节点身份字段。所有模型 metadata、sample rows、normal/SPA state 和 state 声明的数组必须完整列入。

导入工具须完整读取并验证原模型数组 SHA、dtype、shape 与样本行几何，确认全部原拟合进程退出为零，再建立当前运行的链接。清单必须声明 `all_fit_supervisor_exits_zero`、`full_array_hashes_verified`、`full_array_geometry_verified` 为真，`original_model_stores_modified` 为假，`models_refitted` 为零。消费端通过清单 SHA 和未变化的文件身份延续这一完整读取证明，并重新核验 metadata、state SHA；实际加载每个数组时，模型存储读取器还会独立核验该数组 SHA。该设计避免每个 worker 在模型屏障处重复完整扫描所有模型数组。

当前 worker 的完成记录使用 `mode: "verified_reuse"`，分别记录原拟合源码、当前消费源码及两份证明 SHA。记录的 completed 数量表示验证完成的已有模型数。普通拟合计划仍使用当前源码作为拟合来源，记录 `mode: "fresh_fit"`；它不能携带不同的旧拟合来源。

兼容证明不声称新关联计算逐值相同。已有模型的精确拟合状态、协变量与样本轴保持不变；消费者的优化、TF32 舍入及真实关联行为仍需要独立运行验证。局部门禁测试使用匿名文件 fixture，只验证来源、完整覆盖、不可变文件和禁止重新拟合，不代替真实数据 benchmark。

2026-10-11 的资源预算修复允许两种严格证据：原完整 TF32 模块指纹，或 `tf32.py:numerical_component` 指纹加 `tf32_resource_policy_compatibility`。后者明确完整模块已有变化，逐语法节点核验 `configure_tf32` 只新增 reserve 参数、合法性检查、赋值及返回字段，其余 TF32 模块保持一致；它同时绑定此前拟合兼容证明、基线消费源码和当前消费源码。缺少此资源变更证明时，数值组件证据不能单独授权复用。原模型文件、原拟合退出记录、样本轴和当前完整覆盖验证继续强制执行。
