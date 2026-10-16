# GLM-5.2 Ascend 配方迁移上游：实施与验证计划

编写日期：2026-10-16（Asia/Shanghai）

目标仓库：`vllm-project/recipes`

目标页面：[GLM-5.2](https://recipes.vllm.ai/zai-org/GLM-5.2)

文档状态：计划已进入实施；下方记录本轮实际交付及用户确认的执行范围。

### 实施记录（2026-10-16）

用户已确认 tutorials 配置均已验证，要求直接编写 Ascend 增量 YAML 与 Guide，不再进行实机复现和验证；A3、950 使用对应的最新镜像即可。原计划中的硬件复测、完整评测和重新确认支持状态不作为本轮前置条件，历史计划内容保留供后续维护参考。

本轮交付：

- 在 `models/zai-org/GLM-5.2.yaml` 增加 A3 W4A8C8 变体、TP8/DP2/EP、三 token MTP、Ascend parser、ModelScope 权重来源和 `nightly-main-a3` 镜像；硬件验证依据明确引用原教程。
- 既有 Guide 保留，新增完整 Ascend Guide：A2/A3 安装、权重、单/多节点、A3 四节点与 A2 八节点 PD、近 1M 单/双节点与 1M PD、功能请求、原教程精度数据、性能调优及 950 W4A4C8 FAQ。
- 原教程中的 MkDocs tabs/admonition 转为普通 Markdown，相对链接指向固定源提交，镜像占位符展开，Guide 内服务别名统一为 `glm-52`。原始损坏的示例 JSON 不作为可复制响应搬运，改为清晰的响应说明和 chat 请求。
- 命令构建器接入 A3 单节点基线；通用 GPU 多节点/PD 生成器与 Ascend 专用布局不同，其他已验证配置完整保留在 Guide，未新增通用 PD/多节点生成能力或伪装成相同部署。
- 增加硬件范围的 `frontend` 配置，使 Ascend 沿用 Python 前端，同时保留 GPU 的原有默认前端；补齐 Ascend 容器的 `devmm_svm` 设备。

本轮不新增 NPU 测试结果、不重新下载模型权重、不新建 commit/PR。`950` 的源材料包含量化和 C8 配置说明，没有完整 serve 命令，因此按原文纳入 Guide，不自行拼装新的 950 启动配方。

## 1. 结论与交付目标

本次迁移应在已有的 `models/zai-org/GLM-5.2.yaml` 中增补 Ascend 支持。核心工作是把教程中的运行环境、量化权重、并行拓扑和特性约束，转换为上游网站能够正确生成、用户能够复现的配置，同时保留现有 NVIDIA/AMD 路径。

建议按“环境与权重确认 → 上游表达能力补齐 → 基础场景 → 多节点 → PD/长上下文 → 全量验收”推进。可以分批合入，但首次基础场景合入不代表原教程全部迁移完成。未完成场景必须继续列在覆盖矩阵中，直到取得运行证据，或经范围调整明确排除。

最终交付需要同时满足：

1. **内容可追踪**：原教程每个章节、部署场景和重要限制都有迁移位置或保留链接。
2. **数据合法**：YAML 通过上游严格 Schema 和生成检查，不携带下游专用字段。
3. **命令正确**：网页、JSON API、Guide、Verify、Bench 使用一致的环境、权重、拓扑和服务名。
4. **运行可复现**：对公开宣称可运行的组合，提供固定软件/权重版本和目标硬件上的成功推理证据。
5. **不引入回归**：Ascend 参数、镜像和解析器不污染现有 NVIDIA/AMD 配方。

“正确运行”按具体的“硬件 × 权重 × 拓扑 × 上下文 × 特性 × 前端”组合判定，不能由一次启动或一次短问答推导全部组合均已验证。

## 2. 调研依据与版本基线

### 2.1 已读取的材料

| 编号 | 材料 | 用途 |
| --- | --- | --- |
| S1 | [vllm-ascend GLM5.2 原教程](https://github.com/vllm-project/vllm-ascend/blob/932818a5b16dd57bea61a5c2a15c7dc9bda8dea3/docs/source/tutorials/models/GLM5.2.md) | 场景、环境、命令、评测和限制的来源 |
| S2 | [Ascend DeepSeek-V4-Flash YAML 示例](https://github.com/vllm-ascend/vllm-ascend-recipes/blob/d8e9f02d78f30c35b868631303205ab69231dba7/models/zh/DeepSeek/DeepSeek-V4-Flash.yaml) | 参考教程拆解和场景组织；不能直接作为上游模板 |
| S3 | [上游 PR #819](https://github.com/vllm-project/recipes/pull/819)、[最终差异](https://github.com/vllm-project/recipes/pull/819/files) | 参考在既有模型中添加 Ascend 变体、镜像、特性模式和验证说明的方式 |
| S4 | [上游现有 GLM-5.2 配方](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/models/zai-org/GLM-5.2.yaml) | 迁移目标及跨硬件回归基线 |
| S5 | [上游 SCHEMA.md](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/SCHEMA.md)、[机器 Schema 实现](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/src/lib/recipe-schema.js) | 字段、枚举、作用域和合并语义 |
| S6 | [硬件 taxonomy](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/taxonomy.yaml)、[命令生成器](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/src/lib/command-synthesis.js) | 核实设备数、默认命令、镜像选择、并行计算 |
| S7 | [sources.yaml](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/sources.yaml)、[CONTRIBUTING.md](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/CONTRIBUTING.md) | 确认当前配方归属和提交流程 |
| S8 | [HF BF16 config](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json)、[HF FP8 config](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/f33c6dc501ee5a2c7e35155653b1b1abbc320951/config.json)、[模型卡](https://huggingface.co/zai-org/GLM-5.2) | 核对模型身份和架构；不作为 Ascend 量化权重已验证的证据 |

调研时解析到的分支提交：

| 对象 | 提交 |
| --- | --- |
| vllm-project/vllm-ascend main | `932818a5b16dd57bea61a5c2a15c7dc9bda8dea3` |
| vllm-ascend/vllm-ascend-recipes main | `d8e9f02d78f30c35b868631303205ab69231dba7` |
| vllm-project/recipes main | `b3d267608417e41bbbeddccdadb5040b64a8c93f` |
| PR #819 最终 head | `ad311bf113112ff4c83ac4f3d09103fb781d7ab7` |
| 当前本地工作区 HEAD | `03836b4b3c8dbdabc791a7e081b5944bf50beb44` |

本地 HEAD 与远端 main 不同；本次检查的 GLM-5.2 YAML、taxonomy 和机器 Schema 在统一换行符后与上述远端版本一致。实施开始和最终测试前仍须重新对齐目标分支。这里的教程仓库 SHA 是**文档快照**，不是已经验证的运行时 SHA。

### 2.2 本次已核实与尚未核实的边界

已核实：原教程和参考文件内容、PR #819 的合入状态及差异、上游现有配方与相关生成逻辑、HF 两个基础检查点的 config。HF config 均声明 `GlmMoeDsaForCausalLM` / `glm_moe_dsa`，上下文上限为 `1048576`。

尚未核实：Ascend 量化权重的实际可访问性与完整性、适用镜像的存在性及 digest、CANN/驱动兼容性、运行时注册与补丁、真实 NPU 推理、工具调用、MTP、长上下文、PD、精度和吞吐。

本次是计划工作，未下载大型权重、未改写运行配方、未提交 PR。本地尚未安装 `js-yaml` 等项目依赖，因此没有执行配方构建；这不影响文档调研，但后续实施需先准备依赖。

## 3. 两类参考示例应如何使用

### 3.1 Ascend 示例用于拆分内容，不能整体复制

S2 将环境、权重、场景、验证和调优拆开，适合作为内容盘点的参考。但该文件还定义了上游没有的结构，例如 `scenarios`、`config_params`、`weight_download`、`env_setup`，并使用不同的硬件/策略标识；其 `guide` 为空。[来源：S2](https://github.com/vllm-ascend/vllm-ascend-recipes/blob/d8e9f02d78f30c35b868631303205ab69231dba7/models/zh/DeepSeek/DeepSeek-V4-Flash.yaml)

上游机器 Schema 对对象执行未知字段检查。迁移时逐字段转换：运行参数进入上游的 variants/features/overrides，叙述内容进入 `guide`；确需新增能力时同步修改 Schema、文档和消费者。不能把下游文件“能在自己的页面展示”等同于“能通过上游验证”。

### 3.2 PR #819 用于参考增量接入方式

PR #819 已于 2026-08-18 合入，核心方式是在既有模型文件中加入硬件声明、Ascend 专用量化变体、独立 MTP 模式和按精确硬件选择的镜像，同时补充实际验证说明。它还提醒 ModelScope 权重需要明确下载后端或本地路径。[来源：PR #819](https://github.com/vllm-project/recipes/pull/819/files)

GLM-5.2 可以复用这种结构设计，但不能沿用 Qwen 的镜像、MTP 方法、TP 数或验证结论。GLM-5.2 还有跨节点拓扑、PD connector、C8 和 DCP 等额外依赖，必须分别处理。

## 4. 迁移范围与分批交付

### 4.1 建议批次

| 批次 | 内容 | 完成条件 |
| --- | --- | --- |
| M0：基线与缺口 | 固定资料、确定权重/硬件/软件、整理矛盾、生成命令能力评估 | 所有阻塞项有负责人、证据需求和处理路径 |
| M1：基础可运行 | 优先建立 A3 W4A8C8 单节点基线；编码镜像、参数、客户端和特性作用域 | 页面实际复制的命令在目标硬件完成端到端推理 |
| M2：多节点 | A3 双节点、A2 双节点；完善硬件、rank、网络和 headless 行为 | 每个节点生成不同且正确的命令，跨节点推理通过 |
| M3：高级场景 | A3/A2 PD、长上下文、1M 单/双节点及 1M PD；950 系量化路径 | 每个纳入范围的场景都有独立测试报告和上游表达方式 |
| M4：全量收口 | 精度/性能、交互组合、跨硬件回归、上游审查和发布后复核 | 覆盖矩阵闭环；线上页面与已测试配置一致 |

M1 选择 A3 W4A8C8 是基于原教程有完整单节点命令，实际是否可作为首个基线取决于第 6 节阻塞项和硬件资源。若首轮不能得到可靠结果，应更换已取得证据的基线并记录原因。

### 4.2 完整场景覆盖矩阵

以下均为待验证条目；“原教程有命令”不等于本次已经复现。

| ID | 目标场景 | 主要权重 | 迁移方式与验收重点 |
| --- | --- | --- | --- |
| C01 | A3 单节点，常规上下文 | W4A8C8 | 优先验证的基础场景；核对 TP/DP/EP 和容量 |
| C02 | A3 双节点，常规上下文 | W4A8C8 | 本地 DP 数、全局 DP 数、节点起始 rank、每节点 IP |
| C03 | A2 双节点，常规上下文 | W4A8C8 | 确认 A2 真实设备规格、TP8/本地 DP1 拓扑和网络 |
| C04 | A3 四节点 PD | W8A8C8 | P/D 的并行布局、PP 分层、connector 和代理全链路 |
| C05 | A2 八节点 PD | W8A8C8 | 两个池分别分配节点；校验所有 rank 与路由 endpoint |
| C06 | A3 单节点，约 1M | W4A8C8 | 检查 C8+DCP 已知限制后再选配置；实测长输入 |
| C07 | A3 双节点，约 1M | W4A8C8 | 同上，并验证 TP/DP/DCP 的组合与并发容量 |
| C08 | A3 四节点，1M PD | W8A8C8 | 双池启动、权重一致、长输入 KV 传输与输出正确性 |
| C09 | 950PR / 950DT，W4A4C8 | 本地量化产物 | 量化流水线、混合精度、FP8 C8、硬件适用性各自确认 |
| C10 | BF16 / W8A8 权重说明 | BF16、W8A8 | 保留权重信息；若要暴露可运行变体，需补齐独立命令及实测 |
| C11 | 功能、精度、性能、FAQ | 随场景固定 | 迁移评测步骤与限制，原始分数仅标注为来源报告 |

每项最终状态限定为：待澄清、待实现、待验证、验证通过、确认阻塞、经范围调整排除。基础场景成功不得自动更新其余条目。

### 4.3 首轮复现的参数核对表

下表摘取固定教程快照中的关键拓扑值，作为逐项核对的输入，**不是已验证的推荐配置**。特别是 C06/C07，必须先处理 B07 的已知冲突，再决定保留参数、升级运行时或调整权重。

| 场景 | 教程中的关键值 | 迁移时要证明的关系 |
| --- | --- | --- |
| C01 | TP8、DP2、EP；窗口 135000；MTP3 | 实际使用 16 个逻辑设备，DP/EP 与生成器一致 |
| C02 | TP8、全局 DP4、本地 DP2；窗口 66000 | 两节点起始 rank 为 0/2，worker 使用正确 headless 行为 |
| C03 | TP8、全局 DP2、本地 DP1；窗口 40000 | 两节点起始 rank 为 0/1；不按冲突的硬件文字描述推导设备数 |
| C04 | P：TP16/PP2、层分配 41/37；D：DP16/TP2；窗口 202752 | 两个 P 节点、两个 D 节点；检查代理是否覆盖全部 D rank |
| C05 | P：DP4/TP8；D：DP8/TP4；窗口 200000 | P 每节点 1 rank，D 每节点 2 rank，分别使用 4 个节点 |
| C06 | TP16/DP1/DCP16；窗口 1024000 | 真实近 1M 输入、C8/DCP 兼容性及 HBM 余量 |
| C07 | TP8/全局 DP4/本地 DP2/DCP8；窗口 1024000 | 两节点 rank 0/2；窗口和并发分别验证 |
| C08 | P：DP2/TP16；D：DP4/TP8；窗口 1048576 | 每个池 2 节点；connector 声明与实际布局一致 |

在每个场景报告中再补齐 batch tokens、max seqs、内存利用率、图配置、HCCL、parser、dtype、端口和每角色 MTP 深度。不要跨行拼接出原教程未给出、也未经测试的新配置。[参数来源：S1](https://github.com/vllm-project/vllm-ascend/blob/932818a5b16dd57bea61a5c2a15c7dc9bda8dea3/docs/source/tutorials/models/GLM5.2.md)

## 5. 上游目标结构与字段映射

### 5.1 当前归属决策

本次读取的 `sources.yaml` 仍以 recipes 仓库本身为唯一 core source；当前贡献规范要求 Ascend 内容先内联于模型 YAML。因而实施目标是：

```text
models/zai-org/GLM-5.2.yaml          # 既有文件，增量修改
taxonomy.yaml                      # 只有硬件定义确实需要补齐时修改
src/lib/recipe-schema.js            # 新增字段能力时修改
SCHEMA.md                          # 与 Schema 同步
src/lib/command-synthesis.js        # 现有生成器无法表达已验证拓扑时修改
src/components/recipes/CommandBuilder.jsx  # 新能力需要交互或门控时修改
scripts/                           # 必要的命令生成与回归测试
```

不新建 `models/zh/GLM/...`，不把旧 `GLM/*.md` 当作新配方入口；保留 canonical route `zai-org/GLM-5.2`。[当前归属依据：S7](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/CONTRIBUTING.md)

若实施时上游已切换到外部 owner/overlay，再根据最新 `sources.yaml` 调整落点。不能为了未来架构预先把文件放到尚未接入的 source，也不能同时在两个 owner 中创建同一 hf_id。overlay 迁移涉及 add-only 与 Huawei scope，须另行验证覆盖能力。

### 5.2 原教程内容到上游字段

| 原教程内容 | 上游落点 | 实施规则 |
| --- | --- | --- |
| 模型介绍、支持范围 | 保留 `meta`，补充 `guide` | 保留已有模型身份及非 Ascend 内容 |
| 权重来源、下载说明 | `variants.<name>.model_id`、`extra_env`、`guide` | 区分 HF、ModelScope、Modelers；明确本地挂载路径 |
| 量化格式 | variant 的 `precision`、`label`、description | W4A8C8 的格式与 C8 分开说明，不能只写“FP4” |
| 镜像和容器命令 | variant 精确硬件 override + Guide | 对 NPU 固定对应镜像；核对入口命令和设备挂载 |
| HCCL、内存分配等变量 | Ascend variant/hardware/strategy 的 env | 不放到会覆盖 NVIDIA/AMD 的全局 base_env |
| 常规 serve 参数 | variant/hardware/strategy 的 args | 根据作用域拆分，生成器最终值与实测完全对应 |
| 工具/推理解析器 | feature 的硬件 override 或专用模式 | 对齐运行时实现；不能全局替换已有解析器 |
| MTP | `features.spec_decoding.modes`、variant `default_modes` | `deepseek_mtp` 与 `mtp` 的适用性由目标版本实测决定 |
| 单/多节点拓扑 | 上游策略及 `strategy_overrides` | 计算 TP、DP local/global、PP、起始 rank，不能只改策略名称 |
| 1M 和 DCP | opt-in feature、硬件/策略约束、Guide | 明确配置 token 上限及实际测试长度 |
| PD 配置和辅助脚本 | PD overrides、必要的生成器能力、Guide/固定脚本链接 | connector、P/D 参数与路由必须整套迁移 |
| 功能与评测 | `guide` 内的验证、精度、性能章节 | 新运行数据与来源历史结果分开标识 |
| FAQ/已知问题 | `guide` 的 Troubleshooting/References | 保留适用版本、冲突条件及处理方法 |

### 5.3 变体与默认配置设计

建议候选命名为 `ascend_w4a8c8`、`ascend_w8a8c8`、`ascend_w8a8`、`ascend_w4a4c8`；是否全部创建以权重身份和运行证据为准。以上是实施命名建议，不是已经存在的变体。

具体约束：

1. 保留当前 `default` 及既有 FP8/BF16/AMD/NVIDIA 变体；通过 `meta.default_variant` 的硬件映射，为已开放的 NPU 选择合适的 Ascend 变体。
2. 每个 Ascend 变体使用 `supported_hardware` 精确限定 NPU profile。反向检查现有无 allowlist 的变体，避免开放 NPU 后把未经适配的 FP8/BF16 默认提供给 NPU。
3. NPU 镜像优先放在 `variants.<name>.hardware_overrides.<exact_hw_id>.docker_image`；不能假设下游的按 Atlas 型号 image map 可被上游解释。
4. 只有确认没有可运行 pip 安装路径时，才按该 variant+hardware 设置 `install.pip: false`。若提供源码安装，必须固定仓库、SHA、构建依赖和顺序。
5. 用 `meta.hardware` 控制 restricted profile 的可见性；`supported` 不代表实测，`verified` 必须附带具体场景证据。`meta.hardware` 是配方级，Guide 必须补充哪些变体与场景真正测过。
6. 不把未验证解释为 `unsupported`。对已确认会生成错误 connector/拓扑的路径，可以标记该策略当前不可用，并说明阻塞原因；其余通过明确的候选范围和待验证状态管理。

## 6. 实施前必须解决的缺口

### B01：硬件描述和设备计数不一致

原教程对 A3 使用了不同的卡数/内存口径，A2 某处容量描述也与命令的 TP/DP 关系不一致。首先以 `npu-smi`、设备可见性、拓扑和实际执行进程为准建立清单，不能仅按总 HBM 相同认定两个 profile 可互换。[原始材料：S1](https://github.com/vllm-project/vllm-ascend/blob/932818a5b16dd57bea61a5c2a15c7dc9bda8dea3/docs/source/tutorials/models/GLM5.2.md)

当前上游有 `ascend_910c`（16×64 GB），没有下游示例的 `atlas_800_a2` / `atlas_800_a3` 标识；现有 Ascend profile 均未开放多节点。先确认 A3 是否可复用 `ascend_910c`，A2 是否需补充 profile。950 单卡和 8 卡 profile 也不能混用。[代码依据：S6 taxonomy](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/taxonomy.yaml)

关闭条件：形成“物理设备/逻辑设备/单节点设备数/单设备 HBM/互联/型号/profile ID”对照表；新增或开放多节点时完成共享 taxonomy 的影响评估及回归。

### B02：权重身份与下载后端不一致

教程的下载表与部分命令使用不同命名空间；不得推断它们是同一份权重。对每种量化产物获取真实平台、repo、revision、config、量化描述文件、分片清单和 hash。

上游 recipe 的 canonical `model.model_id` 保持 HF 身份；量化 variant 的 `model_id` 是实际加载的 checkpoint 标识，**不自动意味着它存在于 HF**。ModelScope 路径要设置相应后端并验证；Modelers 或本地量化产物若不能由现有 builder/API 表达，列为独立能力缺口，不能虚构 HF 镜像仓库来通过格式检查。

关闭条件：下载、容器内路径、serve、tokenizer、Verify、Bench 五处指向同一产物；本地路径不带出开发机器专用目录；获得完整权重校验记录。

### B03：版本和镜像未固定

教程含文档模板镜像占位符；模型卡、已有配方和不同功能可能要求不同版本。不能拿文档仓库 main 或一个输出 fingerprint 推断所有功能已支持。

需要查明：vLLM 架构注册、Ascend 实现、量化 loader、解析器、MTP、C8/DCP、PD connector 所需最早版本及必要修复。以兼容矩阵固定 vLLM SHA、vllm-ascend SHA、镜像 digest、Torch/torch-npu、CANN、驱动、Transformers、Python/CPU 架构。

关闭条件：镜像可实际拉取、包含目标补丁、入口行为正确，并在每节点记录一致的软件清单；不填写未经验证的“最终推荐版本”。

### B04：全局参数会泄漏到 NPU

当前 GLM-5.2 配方全局带 FP8 KV 参数，默认前端为 Rust；原教程使用的解析器和 MTP 配置与既有配方不同。仅追加 `--quantization ascend` 不足以迁移。[目标配方：S4](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/models/zai-org/GLM-5.2.yaml)

计划逐层检查 base → hardware → variant → strategy → feature → advanced 的最终参数和 env。NPU 按实测覆盖 KV dtype 与解析器，MTP 使用独立模式并限制可选范围。Rust 前端先测；若目标 Ascend 版本无法使用，检查是否需要硬件级前端选择/门控能力，避免全局关闭影响其他硬件。

关闭条件：常用 UI 切换、URL 深链接和 JSON API 都不会遗留不适用的 frontend/feature；返回 NVIDIA/AMD 后恢复原路径。

### B05：混合 TP/DP 和多节点 rank 不能靠重复参数碰运气

按每节点设备数 `G`、每 DP rank 的 TP `T`、同节点 PP=1 时，基本约束为：

```text
dp_local = G / T             # 必须整除；同节点 PP>1 时需重新计入 PP
dp_total = nodes × dp_local
start_rank(node i) = i × dp_local
```

实现前比较生成器与真实 launcher 的约定。当前 multi-node DEP 分支会先生成默认 DP，再处理 overrides；需要用测试证明最终 global/local/rank 三者一致，不能仅发现某个字段能够覆盖就判定通过。单节点 DEP 是否带额外 Router、端口及 HTTP 行为，也要与教程基线对齐。[代码依据：S6 command-synthesis](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/src/lib/command-synthesis.js)

关闭条件：对 2 节点及 4 节点生成全量命令，所有 rank 覆盖正确、不重复、不越界；TP/DP/EP 和实际设备占用一致。

### B06：上游默认 PD 与 Ascend PD 不是同一部署

当前 `pd_cluster` 默认使用 NIXL；教程使用 Ascend 的 Mooncake 路径及专用代理。网页的 Mooncake KV Offload 选项也不能替代 PD connector 迁移。[上游 PD 策略](https://github.com/vllm-project/recipes/blob/b3d267608417e41bbbeddccdadb5040b64a8c93f/strategies/pd_cluster.yaml)

先梳理 connector 的类、模块路径、producer/consumer 角色、KV/控制端口、P/D 并行信息、缓存 dtype、辅助服务及路由协议。A3 常规 PD 含 PP 分层，不能简单套用单一 TP/DEP 模板。需要时新增硬件约束下的 connector/launcher 表达能力，并同步 UI、JSON 和 Schema。

还要核对代理是否覆盖全部 rank：例如教程 A3 常规 PD 部署两个 D 节点、全局 DP16，但所示代理 decoder 列表仅列出 d0 的 8 个端口。需确认是否有额外转发机制；若无，应补全 d1 endpoint 并实测负载分发。不能未经核实认定原代理列表完整。

关闭条件：复制页面完整部署步骤即可启动；客户端请求确实经过 P→KV 传输→D→代理响应，而非仅对 P 或 D 的独立请求成功。必须保留原教程对参考型 PD 部署的生产使用限制。

### B07：长上下文配置与已知限制有冲突

原教程的长上下文命令、调优限制及性能场景表存在需要核对的组合；尤其应核实 W4A8C8 在目标版本的 C8+DCP 条件。不能在冲突未解除时默认开启相关特性。

同时区分 `1024000` 与 `1048576`，以及“模型标称窗口”“服务上限”“实际输入长度”。若调优表的场景名称与窗口值不一致，先由真实脚本、日志和维护者证据澄清。[来源：S1](https://github.com/vllm-project/vllm-ascend/blob/932818a5b16dd57bea61a5c2a15c7dc9bda8dea3/docs/source/tutorials/models/GLM5.2.md)

关闭条件：获得目标版本的支持/修复证据；长输入实际发送成功，并保留输入 token 数、输出预算、并发、峰值 HBM 和结果正确性记录。

### B08：示例 JSON、服务别名和跨仓库链接需修复

原教程的功能输出示例存在格式损坏，且不同场景的服务别名不同。不要逐字搬运响应；使用新生成的有效响应，并在所有入口使用同一服务名约定。

若 builder 默认通过 checkpoint ID 发请求，优先沿用其约定；若必须使用 `--served-model-name`，同时适配 Verify/Bench。`/v1/models` 用于核实实际服务名，权重身份另外由启动日志和 revision 证据确认。将相对文档链接、MkDocs tabs/admonition、Jinja 占位符转换为上游可渲染 Markdown 或固定链接。

关闭条件：客户端和评测命令真实通过；代码块语法检查通过；迁移后的 Guide 无空章节、失效相对路径或未展开占位符。

## 7. 分阶段工作清单

### P0：建立输入清单与范围账本

负责角色：配方负责人、Ascend 运行负责人。

- [ ] 固定 S1/S2/S3/S4 和实现仓库版本，记录文件摘要与检索日期。
- [ ] 将 C01–C11 逐项登记，附原章节、输入文件、预期输出、执行人和状态。
- [ ] 为 B01–B08 建立证据/决策记录，避免在 YAML 中静默修正不确定参数。
- [ ] 清点可用 A2/A3/950 硬件、节点数、共享存储与网络；排定多节点和长上下文窗口。
- [ ] 记录现有 NVIDIA/AMD 命令和关键 JSON，作为后续回归基线。

产物：场景账本、阻塞列表、硬件与版本兼容矩阵。退出条件：首个基线的硬件、权重和安装方式可确定。

### P1：固定运行环境和权重

负责角色：Ascend 运行负责人。

- [ ] 核实宿主驱动/CANN/固件及容器兼容要求，记录 `npu-smi info` 和拓扑。
- [ ] 选择真实存在的镜像，记录 tag、digest、CPU 架构和内部软件版本。
- [ ] 核查模型注册、解析器和关键 Ascend 功能位于所选源码或镜像中。
- [ ] 确认权重平台与 revision，下载后检查每个分片、总大小、hash 和必要量化配置。
- [ ] 各节点使用同一产物及一致容器路径；确认挂载、读权限和存储吞吐。
- [ ] 验证跨节点 HCCL 通信、NIC/IP、路由和端口可达；先通过通信测试再启动大模型。

产物：环境 manifest、权重 manifest、镜像拉取与网络检查记录。退出条件：在指定节点可重复准备完全相同的输入环境。

### P2：先得到成功运行命令

负责角色：Ascend 运行负责人。

- [ ] 从 C01 建立可用配置；保留教程原始参数和每次调整的理由。
- [ ] 初始 smoke 使用保守窗口和并发，明确标注其低于完整目标窗口；不把调小窗口的成功当成原场景通过。
- [ ] 记录权重加载完成、设备使用、编译过程、最终 args/env 和完整日志。
- [ ] 完成健康、模型列表、聊天、流式、工具和推理解析测试。
- [ ] MTP 先独立关闭/开启对照，检查真正产生 draft/accepted 计数，并测正确性。
- [ ] 按矩阵恢复目标窗口/并发；确认优化项未覆盖其他 JSON 配置或引入冲突。

产物：可以独立复现的启动脚本和端到端结果。退出条件：获得基线运行证据；此时才把最终命令编码成 YAML。

### P3：补齐上游表达能力

负责角色：配方负责人；涉及公共代码时由渲染器维护角色参与。

- [ ] 根据 B01 补充或复用硬件 profile，限定 restricted 可见性和多节点能力。
- [ ] 验证现有 `tp`、`default_variant`、`strategy_hardware`、feature hardware/mode 等字段是否足够。
- [ ] 检查 NPU 镜像/设备数、安装 Tab、命令入口及共享目录生成。
- [ ] 用实际拓扑构造单节点 DEP、多节点 DEP 的参数测试，检查去重后 global/local/rank 的值。
- [ ] 若需要前端硬件门控、PD connector/launcher 或本地权重来源能力，独立设计最小字段并同步 Schema、SCHEMA.md、UI、JSON 消费者。
- [ ] 对公共代码补充有意义的正反例和跨硬件回归；不依赖肉眼判断命令字符串。

产物：必要的基础设施改动及测试。退出条件：已验证场景可以由上游数据模型完整表达；不能表达的场景继续作为阻塞项，不能仅在正文留一个命令就标为完成。

### P4：编写 Ascend 增量配方与 Guide

负责角色：配方负责人。

- [ ] 在现有 YAML 增加已具备证据的变体、硬件状态、镜像和模式。
- [ ] 将不同硬件/场景的参数放入正确 override；避免全局 NPU env 或解析器污染。
- [ ] 一个 `--additional-config` / `--compilation-config` / `--speculative-config` 对应可解释的最终对象；验证后层覆盖是否整体替换，不能假定深合并。
- [ ] feature 关掉后对应可选参数能真正消失；必要参数不能由开关误删。
- [ ] 按第 8 节结构编写英文上游 Guide；当前中文计划保留为工作文档。
- [ ] 把未测试窗口、功能、量化格式和历史性能结果明确区分，补充固定来源链接。

产物：可构建的增量 YAML、完整 Guide。退出条件：生成命令与 P2 实测命令语义一致。

### P5：多节点与高级场景逐项复现

负责角色：Ascend 运行负责人、配方负责人。

- [ ] C02/C03：逐节点运行生成命令，核对 rank、HCCL、API/headless 和模型加载。
- [ ] C04/C05：分 P/D 池逐步启动，检查全量 endpoint、KV 传输与代理行为。
- [ ] C06/C07：解决已知冲突后递增长输入，直至约定的近 1M 上限及并发。
- [ ] C08：执行长输入 PD，确认读取远端 KV，不能只做短 prompt。
- [ ] C09：固定 msmodelslim 版本、量化配置及校准数据版本，再分别测试 950PR/950DT；不能共享一次验证结论。
- [ ] C10：如果对外开放 BF16/W8A8 运行变体，补齐独立实测；否则仅保留准确的权重来源说明。
- [ ] 结果回写场景账本，失败项记录确切错误、适用版本、重试条件和修复归属。

产物：每场景启动脚本、请求/响应、日志和验证报告。退出条件：约定范围内的 C01–C11 均有明确结论和对应交付位置。

### P6：评测、页面验收和提交

负责角色：验证人员、配方负责人。

- [ ] 使用页面复制出的最终命令再跑一次关键基线，固定最终测试 SHA。
- [ ] 执行第 9 节的静态、页面、API、推理、精度、性能和回归验证。
- [ ] PR 附覆盖矩阵、环境/权重指纹、成功场景及剩余限制；不把历史原文结果当成本次成绩。
- [ ] 如公共能力尚未合入，先合入基础设施 PR，再提交依赖它的配方 PR。
- [ ] 按 DCO 使用 `git commit -s`；只提交源文件，不提交 `public/`、`.content/`、权重、缓存或本地环境配置。
- [ ] 合入后检查线上 route、JSON、链接和命令，与测试产物比对；最后更新原教程的迁移入口。

产物：可审查 PR、验证证据、上线核对记录。退出条件：第 12 节完成定义全部满足。

## 8. 上游 Guide 建议目录

```text
Overview                         保留现有模型介绍，说明 Ascend 增量支持范围
Ascend prerequisites             硬件、版本矩阵、镜像、网络、存储
Ascend checkpoints               平台、revision、量化格式、下载和路径
Ascend installation              每类硬件的容器与必要安装步骤
Ascend single-node serving       最小基线和完整目标配置
Ascend multi-node serving        每个节点独立命令与 rank/端口表
Ascend PD disaggregation         参考部署限制、P/D/代理/connector 全流程
Ascend long-context serving      窗口、DCP/C8 约束、资源要求
Client and functional checks     health/models/chat/stream/tools/reasoning
Accuracy and performance         本次实测、方法、来源历史数据
Troubleshooting                  环境/加载/网络/解析器/容量/PD 问题
References                       原教程、精确权重、运行时和评测文档
```

不以迁移新目录为理由删除已有 GPU 章节。保留历史教程链接，在高级场景尚未迁移完成时仍能找到完整原说明；但不能用一个链接替代已承诺的 builder/运行迁移交付。

## 9. 验证方案与通过标准

### 9.1 静态与构建检查

以下命令在后续实施时运行，本次未执行。先在 recipes 工作区安装锁定依赖：

```bash
pnpm install --frozen-lockfile
pnpm test:schema
node scripts/build-recipes-api.mjs
git diff --check
```

如修改 overlay/source/合并能力，再执行：

```bash
pnpm test:overlay
```

若未来配方落在外部 source，再根据实际 source 名称和本地 checkout 运行：

```bash
node scripts/validate-source.mjs --source <configured-source> --dir <recipe-root>
```

该命令中的 source/path 必须来自最新 `sources.yaml`，不能机械填写 `vllm-ascend`。现阶段直接内联修改用常规 JSON 构建即可。涉及 PD/路由公共逻辑时，额外运行对应已有测试及本次增加的拓扑测试。

通过标准：无未知字段和非法硬件/策略 ID；生成的模型 JSON 包含预期变体、来源、硬件和命令；所有预期文件均生成，未开放组合没有被意外输出为已支持路径。

不以反复运行 `pnpm build` 替代上述检查，也不终止/重启已有开发服务器。若已有 `pnpm dev`，使用当前服务刷新预览；没有时才启动一次。

### 9.2 命令与页面检查矩阵

| 维度 | 检查内容 | 通过条件 |
| --- | --- | --- |
| Hardware | NPU 型号、设备数、总内存、单/多节点能力 | 与实际机器及 profile 一致 |
| 默认状态 | 选择 NPU 后的默认 variant、strategy、frontend、features | 得到已测基线；无 GPU 默认参数残留 |
| Variant | 权重、量化 flag、容量提示、下载后端 | 选中权重与实际加载产物一致 |
| Install | Docker/Pip Tab、镜像、入口、设备/驱动/权重挂载 | 可直接执行；不回退到 CUDA 镜像 |
| Strategy | TP/DP/EP/PP、节点数、rank、端口 | 与真实拓扑一致，每节点命令完整 |
| Feature | 工具、推理、MTP、DCP/C8 的组合与开关 | 参数可加入/撤销；无不适用模式 |
| JSON 参数 | 多层 additional/compilation/speculative config | 最终对象无意外丢字段或冲突覆盖 |
| PD | connector、角色、代理、endpoint、启动次序 | 复制全部步骤可完成 P→D 请求 |
| Verify/Bench | base URL、model/served name、tokenizer | 与启动配置一致并实际通过 |
| 分享链接 | URL 指定 hardware/variant/feature/frontend | 无效状态被正确处理，不绕过硬件门控 |
| 切换回归 | NPU→NVIDIA→AMD→NPU | 不遗留上一硬件的 env、镜像、feature 模式 |
| Guide | Markdown、代码块、完整链接 | 无 MkDocs/Jinja 残留和跨仓库相对链接错误 |

检查选中组合的最终 `argv`/env 和页面文本两部分；字符串看起来相似不证明 shell 执行语义一致。特别检查空格、引号、JSON 和变量展开。

### 9.3 基础运行验收

| 用例 | 必须记录 | 判定 |
| --- | --- | --- |
| 启动与加载 | 运行时 SHA、权重 revision、解析架构、各卡内存、最终参数 | 所有预期进程就绪，无未知参数/加载失败 |
| `/health` | HTTP 状态及时间 | 成功返回 |
| `/v1/models` | 返回的 model ID / 服务别名 | 与 recipe/客户端约定一致 |
| 非流式 chat | 原始请求/响应、content、usage、finish_reason | 产生合理非空最终回答；基础 stop 用例正常结束 |
| 流式 chat | SSE 片段、最终终止信息 | 能正确重组最终回答，无卡住和不完整 JSON |
| 来源指纹 | system_fingerprint（如提供）、启动日志、镜像/源码信息 | 能关联到目标构建，不能只凭模型列表确认来源 |
| 负向检查 | 错误 model ID、超出窗口、非法特性组合 | 合理错误或 UI 阻止，不静默切换其他权重 |

基础 chat 用充足输出预算与能明确终止的任务验证 `finish_reason: stop`；不能把被截断的 `length` 响应当作完整结束。工具调用用例可合法返回 `tool_calls`，不强制使用 stop；其后须完成工具结果回传和最终回答。

### 9.4 功能专项验收

| 功能 | 用例与证据 | 禁止的推断 |
| --- | --- | --- |
| 工具调用 | 一个简单确定性工具，检查工具名/JSON 参数/结果回传/最终回答；补一条流式工具用例 | 带 parser 参数启动成功不代表工具调用成功 |
| 推理解析 | 用可检查结果的任务，核对最终 content 与推理字段的接口行为 | 不能仅看输出包含 think 字样 |
| MTP | 同环境关闭/开启对照；draft/accepted 指标、结果、延迟和显存 | 启动日志提到 MTP 不证明运行中有效 |
| C8 | 分别验证 attention KV dtype 和 indexer dtype | 名称含 C8 不证明两个缓存都使用同一格式 |
| 长上下文 | 逐步提升真实输入；报告 tokenizer 计数、窗口、输出预算、并发、HBM、正确性 | `max-model-len` 设置为 1M 不证明处理过 1M |
| 多节点 | 全节点日志、实际 rank、通信、客户端结果、超时排查 | 只检查 head ready 不代表 worker 正常 |
| PD | 代理请求、P/D 两侧日志、KV 传输、无意外完整重算；检查错误和超时行为 | P/D 各自 health 正常不代表 PD 链路成功 |
| W4A4C8 | 量化配置/校准记录/产物校验，按 950 型号复现正确性与精度 | 不能按其他 INT4/NVFP4 变体推断其支持范围 |

### 9.5 精度与性能评测

精度按原教程使用的 AISBench 路径复现，但在测试前固定工具版本、数据集版本、样本数、prompt/chat template、采样配置、输出长度和种子。先测短小的确定性正确性集合，再测完整目标集。必要时多次运行报告均值和波动。

验收阈值须在运行前由负责方确定，写入场景报告：绝对分数下限、相对可信基线允许的下降百分点、运行次数。若缺少匹配版本和配置的基线，先建立基线，不能直接把原教程某个历史分数当硬门槛。

性能同时记录输入/输出 token 分布、并发/RPS、请求总数、成功/失败数、TTFT/TPOT/ITL 分位数、输出吞吐、总吞吐、峰值 HBM，以及 MTP 指标（适用时）。区分冷启动、编译预热和稳态。禁用/启用 prefix cache 的状态必须固定，真实评测负载与 synthetic 负载分别说明。

通过标准：预先约定负载下无失败/异常 OOM，满足已登记的性能目标和精度阈值；报告足以让其他人复跑。没有匹配实测数据时不新增吞吐 headline，也不沿用其他硬件或其他模型的数字。

### 9.6 现有 NVIDIA/AMD 回归

至少比较修改前后的 H200 默认 FP8、B200/B300 NVFP4、AMD MI355X MXFP4 及既有 MTP 模式的生成命令、镜像、默认前端、Verify/Bench ID 和可见策略。语义不应因 Ascend 增量发生意外变化。

若改公共生成器、全局参数或 frontend 行为，执行相应硬件上的代表性运行回归；若当前无资源，报告静态验证范围，并把所需硬件运行回归列为合入/完整验收的待办，不能写成已通过。

## 10. 证据包与可复现记录

建议每场景形成独立目录或 CI artifact，较大的日志与结果放在可访问的长期存储中，PR 提供链接。不要将权重、模型缓存或全部测试输出提交 recipes。

```text
<scenario-id>/
  manifest.yaml                环境、硬件、镜像、源码、权重和评测版本
  launch-node-0.sh              实际运行的命令
  launch-node-1.sh              多节点场景逐节点保留
  requests/                    原始测试请求
  responses/                   合法且脱敏的原始响应
  logs/                        启动、运行、P/D/代理和异常日志
  benchmark-results.json       机器可读结果
  report.md                    结论、限制、通过/失败项及证据链接
```

manifest 至少记录：

| 分类 | 字段 |
| --- | --- |
| 场景 | ID、测试日期、测试人员、通过标准、实际状态 |
| 硬件 | 型号、物理/逻辑设备数、单卡/节点 HBM、节点数、拓扑、NIC |
| 软件 | 镜像 tag/digest、vLLM/Ascend SHA、Torch/torch-npu、CANN/驱动、Transformers、Python |
| 权重 | 平台、完整 ID、revision、分片数/总字节/hash、量化配置、tokenizer |
| 运行 | 最终 args/env、端口、rank 表、frontend、feature 开关、窗口/并发 |
| 结果 | health/models/chat/专项结果、fingerprint、精度、性能、峰值 HBM、失败原因 |
| 来源 | 原教程快照、上游 recipe SHA、页面复制命令或对应 JSON 摘要 |

对外共享前脱敏内部 IP、账户、token 和敏感路径；保留能复现的占位符定义与填写规则。

## 11. PR 拆分、资源与时间安排

### 11.1 PR 拆分建议

| PR | 主要内容 | 依赖/审查点 |
| --- | --- | --- |
| PR-A（必要时） | 硬件 profile、多节点/混合拓扑、前端/connector 等公共能力 | Schema/文档/生成器/消费者/回归一起提交；不直接加入虚假的 verified |
| PR-B | GLM-5.2 Ascend 单节点基线、变体、镜像、功能与 Guide | 依赖 PR-A 中真正需要的能力；附 C01 证据 |
| PR-C | A3/A2 多节点与 PD | 依赖已验证的网络、拓扑和 connector；附 C02–C05 |
| PR-D | 长上下文、1M PD、950 量化和评测收口 | 附 C06–C11；关闭覆盖矩阵剩余项 |
| PR-E（后续源仓库） | 原教程入口、链接和维护归属说明 | 上游合入且线上复核通过后再调整入口 |

实际可合并相近 PR，但避免让尚未验证的高级场景阻塞一个已有充分证据的基线。每个分批 PR 清楚说明当前交付范围和后续项；全项目完成仍以完整范围账本为准。

本次只交付计划，不创建分支、commit 或 PR。后续实现分支可采用 `codex/glm52-ascend-migration`，提交遵循 sign-off。

### 11.2 角色与资源

| 角色 | 责任 | 资源要求 |
| --- | --- | --- |
| 配方负责人 | 字段转换、Guide、UI/JSON 一致性、PR | recipes 开发环境 |
| Ascend 运行负责人 | 环境/权重、基线、多节点、PD、长上下文 | 对应 A2/A3/950 硬件和存储 |
| 验证负责人 | 独立执行、精度/性能报告、回归 | AISBench/数据集、硬件测试窗口 |
| 上游维护者 | 公共模型/Schema/生成器设计审查、合入 | 评审与 CI |

多人可由同一人承担；这里是职责划分，不要求额外创建代理或聊天任务。

### 11.3 工作量估计

以下为计划估计，假设硬件、镜像、权重和评测数据均可获得；不含机器排队、TB 级权重传输、长时间完整评测和社区 review 等待。

| 工作 | 估计人日 | 主要不确定性 |
| --- | --- | --- |
| 基线盘点、矛盾澄清、兼容矩阵 | 1–2 | 权重命名与硬件口径 |
| 环境/权重准备、首个成功基线 | 1–3 | 镜像/内核/量化支持 |
| 上游字段转换、基础 Guide 与单节点验收 | 1–2 | 默认参数、frontend 和功能门控 |
| 公共能力补齐（如需） | 2–5 | 多节点、PD、Schema 变更范围 |
| A2/A3 多节点及 PD 复现 | 2–4 | HCCL、PP 拓扑、connector/代理 |
| 1M 与 950 量化专项 | 2–5 | 已知冲突、长上下文容量、量化流程 |
| 评测、跨硬件回归与审查收口 | 2–3 | 数据集耗时、运行差异 |

总体约 **11–24 人日**，属于当前缺口下的估计，不是承诺工期。基线工作和基础设施分析可在职责明确后并行；高级场景必须依赖已确认的软件和权重。P0 完成后应据实际资源重新估时。

## 12. 完成定义与最终检查表

### 12.1 单批 PR 可合入条件

- [ ] 本批公开暴露的硬件、权重、策略、frontend 和 feature 组合与测试范围一致。
- [ ] 所有必填字段有来源，版本/镜像/权重可复现，没有猜测性的 HF ID 或镜像标签。
- [ ] Schema/JSON/必要公共逻辑测试通过，页面和 API 与实测命令一致。
- [ ] 有目标硬件上真实生成最终回答的证据，附带模型身份和运行时指纹。
- [ ] 已确认不适用的组合被准确门控；未验证项不标 verified。
- [ ] 相关 NVIDIA/AMD 路径完成约定回归；未覆盖部分明确列出。
- [ ] Guide 包含适用范围、已知限制、有效权重链接和可复现步骤。
- [ ] 只提交预期源文件；通过 `git diff --check`；commit 带 DCO sign-off。

### 12.2 GLM-5.2 整体迁移完成条件

- [ ] C01–C11 全部闭环，原教程每个章节均在映射账本中有归宿。
- [ ] B01–B08 已关闭，或经明确范围调整记录排除项及理由。
- [ ] 所有对外宣称可运行的场景均有实测报告，PD 和近 1M 输入分别验证。
- [ ] 精度/性能结果使用固定方法和可接受阈值，不混用原文历史数据。
- [ ] 上游 PR 已合入，线上模型页/JSON/命令已核对。
- [ ] 源教程保留必要历史信息，迁移入口和后续维护归属清晰。

若仅完成 C01，应报告“GLM-5.2 Ascend 单节点支持已接入，完整迁移继续进行”。只有满足本节整体条件，才报告原教程迁移完成。

## 13. 批量迁移其他模型时复用的流程

GLM-5.2 完成后，可以复用以下固定流程，但每个模型仍需独立运行证据：

```text
固定源教程与上游版本
  → 查找现有 canonical recipe 与当前 owner
  → 盘点章节、权重、硬件、拓扑、功能及冲突
  → 验证镜像/源码/权重身份
  → 取得真实成功命令
  → 转换为上游字段并补齐必要公共能力
  → 比对网页、JSON、Guide、Verify、Bench
  → 目标硬件复现与专项/回归评测
  → 分批 PR、DCO、线上核对
  → 更新源文档入口和长期维护规则
```

建议把场景账本、环境/权重 manifest、命令比对和验收报告沉淀成模板。自动化可以辅助抽取命令和检查字段，但不能用 YAML 转换成功代替真实 NPU 推理，也不能用一个模型成功代替另一个模型的验证。
