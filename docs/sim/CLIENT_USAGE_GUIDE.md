# G1+O6 / G1+Wuji 仿真闭环 — Client 使用速查（最新）

> 本文是**最新**的客户端使用入口。深入主题（协议、重启、场景切换、layout 切换、批量评测）见各专题 doc，本文只做速查 + 端到端示范。
>
> 适配的服务端版本：`source/easim/scripts/scene_ws_action_server_g1o6.py`。当前包含 25 个 scene ID、`arm26` / `full36` / `wuji54` / `r1pro32` action layout、单目/双目/四相机/LiDAR 观测、rollout 泛化和 server 端评测；部署在 `172.31.0.226`，pool 监听 `ws://172.31.0.226:8080–8083`。R1 Pro client 请直接阅读 [`R1_PRO_CLIENT_USAGE_GUIDE.md`](R1_PRO_CLIENT_USAGE_GUIDE.md)。

---

## 0. 一分钟速览

```
client                                                            sim host
  │                                                                   │
  │  1) 选 worker（端口）+ probe status                                │
  │  2)  按需 switch_scene / switch_action_layout / switch_scene_and_layout │
  │      / switch_pick_fruit_layout                                         │
  │      （独占；触发 server 自重启）                                        │
  │  3)  subscribe_step_result -> 如有需要先 reset，再回 initial frame    │
  │  4)  loop: submit_actions(N×D) -> ack -> step_result(post_actions)│
  │  5)  下一次 rollout：再次 subscribe_step_result 或 reset_env       │
  │  6)  断连或 unsubscribe                                            │
```

- **D = 26**（arm26）、**36**（full36）、**54**（wuji54）或 **32**（r1pro32）。`status_response.action_dim` 是真值；O6 正式 worker 通常为 `arm26 / 26`，G1+Wuji 场景必须为 `wuji54 / 54`，R1 Pro 场景必须为 `r1pro32 / 32`。
- 一个 worker 同一时刻**只服务一个 client**；多并发会互相覆盖
- **switch_scene / switch_action_layout / switch_scene_and_layout / switch_pick_fruit_layout 都会让 server 自重启**（~45–90s 不可用），独占语义；跨 O6/Wuji/R1 Pro 必须用 `switch_scene_and_layout` 一次同时切 scene 与 layout。


### 0.1 Server 功能总览

| 版本 | 模块 | 当前支持内容 | 说明 |
|---|---|---|---|
| v1.4 | 支持场景 | `pick_fruit` 抓水果场景 | 主要水果场景。支持固定三水果、随机三水果、单水果、负样例、自定义初始状态、机器人手臂/手指初始关节泛化；当前资产配置支持 100 种水果 leaf，评测读取运行时 PhysX/world pose。 |
| v1.5 | 支持场景 | `pick_fruit_wuji` G1+Wuji 抓水果场景 | 与 `pick_fruit` 使用同一份水果 USD、对象泛化和 PhysX 评测，只替换为 G1 + 双 Wuji 20-DoF 灵巧手。必须使用 `wuji54`；输出 54 维 action、69 维全身状态和 40 维双手状态。机器人关节泛化会匹配 14 个双臂关节和 40 个 Wuji 手关节，不随机 root。 |
| v1.4 | 支持场景 | `pick_fruit_office1` Office1 抓水果场景 | 独立使用 `office_1.usd` 和对应桌前机器人位姿；复用 `pick_fruit` 的 rollout 泛化、负样例、自定义初始状态和运行时 PhysX 评测。当前 USD 可管理 `qsim_pumpkins`、`qsim_carrot`。 |
| v1.4 | 支持场景 | `pick_fruit_lidar` LiDAR 抓水果场景 | 与 `pick_fruit` 共用水果场景、泛化和运行时 PhysX 评测；额外创建与 `viewport_cam` 共位的 RTX LiDAR，并支持显式开启点云调试录制。 |
| v1.4 | 支持场景 | `pick_fruit_stereo` 双目抓水果场景 | 与 `pick_fruit` 共用水果场景、泛化和评测逻辑；额外返回左/右 RGB 图片，左目继续兼容旧 `image` 字段，右目通过 `images.right` 读取。当前左右目挂载在机器人 `head_link` 下，左目按 `T_world2cam` 中的 pelvis→left camera 外参渲染，右目由 stereo `R/T` 推导。 |
| v1.4 | 支持场景 | `water_plant` 浇水场景 | 支持水壶和花盆的 rollout 级泛化；client 可指定对象，也可由 server 随机对象种类、位置和 pose。 |
| v1.4 | 支持场景 | `stack_plate_bowl` 盘子碗场景 | 支持默认 2 plate + 1 bowl、随机 2 个不同 plate + 1 bowl、指定对象模式；支持桌面 XY 和 yaw 泛化。当前只接入场景/泛化，未接入自动评测 adapter。 |
| v1.4 | 支持场景 | `pick_paper_balls` 抓纸团/果皮投垃圾桶场景 | 支持橘皮、纸团、垃圾桶的 rollout 级泛化；支持默认、随机、固定对象、自定义初始状态。自定义模式可指定不同种类和位置，数量上限为资产池种类数，同类 leaf 不重复。 |
| v1.4 | 支持场景 | `pick_fruits_and_paper_balls` 水果+纸团/果皮组合整理场景 | 组合场景：每个 rollout 包含 3 个水果 + 1 个碗/盘 + 1 个果皮 + 1 个纸团 + 1 个垃圾桶；支持 `default` / `random` 两种 rollout 级泛化。当前只接入场景/泛化，未接入自动评测 adapter。 |
| v1.4 | 支持场景 | `pick_fruits_and_paper_balls_showroom` Showroom 复杂桌面整理场景 | 独立使用 `scene_1.usd` 和桌前机器人位姿；固定显示 3 个水果、碗、垃圾桶、纸团和果皮；当前使用 USD 原始布局，不复用 Room01 随机泛化。 |
| v1.4 | 支持场景 | `pour_coffee` 倒咖啡场景 | Office Room01 Area_9 场景，支持 1 个 source container + 1-2 个 target cup；支持默认、随机、固定对象，以及 client 指定 kettle/cup 桌面 XY 的 custom 模式。当前只接入场景/泛化，未接入自动评测 adapter。 |
| v1.4 | 支持场景 | `pour_coffee_stereo` 双目倒咖啡场景 | `pour_coffee` 的双目 RGB 版本；复用同一套 Area_9 场景和泛化逻辑，左目兼容旧 `image` 字段，右目通过 `images.right` 读取；相机挂载和标定与 `pick_fruit_stereo` 保持一致。 |
| v1.7 | 支持场景 | `showroom_scene_2/3/4/7/8/9/10/11` | 独立加载 Showroom 对应资产，沿用 G1+O6 `arm26/full36` replay 与 RGB 观测协议，不复用 Room01 场景随机化。scene_11 新增单字母/13 字母 rollout 泛化；scene_10 当前使用 `Rubiks_Cube_physics.usdz`。 |
| v1.7 | 支持场景 | `showroom_scene_11_wuji` | 与 `showroom_scene_11` 使用 `Showroom_standalone/scene_11/scene_11.usd` 和完全相同的字母 rollout 泛化，机器人替换为 G1 + 双 Wuji 20-DoF 灵巧手；必须使用 `wuji54`，输出 54 维 action、69 维全身状态和 40 维双手状态。 |
| v1.6 | 支持场景 | `laundryroom_r1_pro` R1 Pro 洗衣场景 | 独立使用 `Laundryroom/Laundryroom_v1_0.usd`。先加载房间并停用 USD 内嵌 R1，再独立生成 IsaacLab 管理的固定基座 R1 Pro；输出头部左右目和左右腕四路 640x480 RGB。必须使用 `r1pro32`；一阶段执行双臂/夹爪，torso 与机器人站位可由 client 按 rollout 配置，忽略底盘与 padding。 |
| v1.4 | 支持场景 | `pick_bowl` 单碗场景 | 基础默认场景，用于简单抓取和动作链路验证。 |
| v1.4 | 支持场景 | `pick_cereal_box_and_milk` 牛奶盒/谷物盒场景 | 桌面物体抓取场景，支持通过 `switch_scene` 切入。 |
| v1.4 | 支持场景 | `open_fridge_pick_milk` 冰箱取牛奶场景 | 办公室冰箱取物场景，支持通过 `switch_scene` 切入。 |
| v1.7 | Rollout 泛化 | rollout 边界重置与对象泛化 | 5 个水果 scene ID、`water_plant`、`stack_plate_bowl`、`pick_paper_balls`、`pick_fruits_and_paper_balls`、`pour_coffee` / `pour_coffee_stereo` 及两个 Scene11 scene ID 可在 `subscribe_step_result` 或 `reset_env` 时应用各自布局；Scene11 支持单字母/13 个不重复字母的种类、位置、颜色可选泛化，不做大小泛化。各模块不需要 hotkey，也不共享泛化状态。 |
| v1.4 | 自动评测 | 单水果评测 | 适用于 `pick_fruit` / `pick_fruit_wuji` / `pick_fruit_office1` / `pick_fruit_lidar` / `pick_fruit_stereo`。判断唯一 active fruit 是否最终进入碗内，输出 `metrics.pass`；多 rollout 汇总为 `pass_rates.pass_rate`。 |
| v1.4 | 自动评测 | 多水果评测 | 适用于上述 5 个水果 scene ID。判断至少 1/2/3 个水果是否进入碗内，输出 `pass_at_1`、`pass_at_2`、`pass_at_3`；多 rollout 汇总为对应通过率。 |
| v1.4 | 自动评测 | 纸团/果皮投垃圾桶评测 | 判断所有 active 橘皮和纸团是否最终进入任一 active 垃圾桶，并且所有 active 垃圾桶没有被打翻；输出 `metrics.pass`，多 rollout 汇总为 `pass_rate`。 |
| v1.4 | 自动评测 | Debug 评测信息 | 可选开启 `debug_details=true`。水果评测返回水果/碗位姿、距离阈值、失败原因等；纸团/果皮评测额外返回垃圾桶姿态、`container_pose_ok`、`containers_not_upright` 等信息。 |
| v1.4 | 结果落盘和平台展示 | 评测任务落盘 | server 自动记录任务基本信息、summary、每个 rollout 的 report；client 不需要自己汇总多 rollout 通过率。 |
| v1.4 | 结果落盘和平台展示 | 图片/视频 artifacts | 默认保存每个 rollout 的首帧和末帧图片；视频默认关闭，可通过 `record_artifacts.video=true` 显式开启。 |
| v1.4 | 结果落盘和平台展示 | 共享存储 | 数据统一落盘到 `/kairos_vepfs_volc/simulation/xuqiang/easim_platform_data`，可用 `EASIM_PLATFORM_RECORD_ROOT` 覆盖。 |
| v1.4 | 结果落盘和平台展示 | 仿真平台展示 | `easim_platform` 读取共享存储，展示任务列表、任务信息、结果统计和 rollout 详情。 |
| v1.6 | 控制与协议 | `arm26` / `full36` / `wuji54` / `r1pro32` action layout | O6 场景使用 `arm26` 或 `full36`；G1+Wuji 场景固定使用 `wuji54`；R1 Pro 场景固定使用 `r1pro32`。`status_response.action_dim` 是 client 对齐动作维度的真值。layout 切换会触发 worker 自重启。 |
| v1.6 | 控制与协议 | 场景/layout 切换与状态查询 | 支持 `status`、`switch_scene`、`switch_action_layout`、`switch_scene_and_layout`、`switch_pick_fruit_layout`、`reset_env`、`subscribe_step_result`、`submit_actions`、`start_evaluation` / `finish_episode` / `finish_evaluation` 等 websocket 协议。`switch_scene_and_layout` 原子支持 O6、Wuji、R1 Pro 之间切换。 |

---

## 1. 当前可用的 worker、scene、layout

### 1.1 Pool worker

| Worker | 固定端口 | 运行时 scene/layout |
|---|---|---|
| 0 | 8080 | 可变，连接后查询 `status` |
| 1 | 8081 | 可变，连接后查询 `status` |
| 2 | 8082 | 可变，连接后查询 `status` |
| 3 | 8083 | 可变，连接后查询 `status` |

端口是稳定入口，scene、robot、layout 和 active objects 都是运行时状态，可能被 operator 或前一个独占 client 切换。每次连接必须先发 `{"type":"status"}`，至少核对 `scene_id`、`robot`、`action_layout`、`action_dim` 和 `step_result_state_dim`。Per-worker 状态持久化在 `/tmp/g1o6_server_scene_id_<port>`、`/tmp/g1o6_action_layout_<port>` 和 `/tmp/g1o6_pick_fruit_layout_<port>`；代码级默认 `pick_bowl / full36` 只在对应状态文件不存在时生效。

当前部署的 `/root/code/easim/assets` 最终解析到 `/kairos_vepfs_volc/simulation/xuqiang/easim_assets/releases/assets_20260803`。部署或排查 USD 差异时先执行 `readlink -f /root/code/easim/assets` 核对；如果软链接切换过，必须完整重启对应 worker 后再 smoke。

### 1.2 可用 scene_id

| `scene_id` | 内容 |
|---|---|
| `pick_bowl` (代码默认) | 单碗在桌面 |
| `pick_cereal_box_and_milk` | 桌面 + 牛奶盒 + 饮料盒 |
| `open_fridge_pick_milk` | 办公室 + 预开门冰箱 + 牛奶盒（牛奶 off-world x≈1000）|
| `pick_fruit` | Office_10F_Room01 + 红碗 + `pick_fruits_assets.yaml` 中配置的 100 种水果；当前 Office USD 只解析已有 prim，缺失资产会跳过，支持 rollout 级水果组合/位置/pose 泛化和负样例 |
| `pick_fruit_wuji` | `pick_fruit` 的隔离 G1+Wuji 版本；使用同一份 `Office_10F_Room01_pick_fruits.usd`、默认机器人 root `(0.35, 1.80, 0.76)`、水果泛化、负样例、自定义布局和运行时 PhysX 评测。必须使用 `action_layout=wuji54`。 |
| `pick_fruit_office1` | 独立使用 `Office_10F_Room01/office_1.usd`；机器人默认 env-local root pose 为 `(-0.2513279542, 2.7930426597, 0.76)`、yaw 约 30°。复用水果 rollout/负样例/custom/PhysX 评测逻辑；当前 USD 可解析的受管水果为 `qsim_pumpkins`、`qsim_carrot`，因此应显式使用单水果模式。 |
| `pick_fruit_lidar` | `pick_fruit` 的 RTX LiDAR 版本；复用水果 USD、rollout 泛化和 PhysX 评测，额外提供与 `viewport_cam` 共位的 LiDAR 传感器及可选点云调试录制。 |
| `pick_fruit_stereo` | `pick_fruit` 的双目 RGB 版本；左目 sensor key 为 `viewport_cam`，右目 sensor key 为 `viewport_cam_right`，图片尺寸按当前标定为 640×480；相机挂载在 `head_link` 下；`status_response.stereo_calibration` 返回 K/D/R/T/R1/R2/P1/P2/Q、左目 `T_world2cam`、渲染用左右目 pose 等 metadata |
| `water_plant` | Office_10F_Room01 Area_3 浇水场景；client 可指定水壶和花盆，也可在每个 rollout 随机对象种类、位置和 pose。 |
| `stack_plate_bowl` | Office_10F_Room01 Area_6 + `stack_plate_bowl_assets.yaml` 中的 6 种 plate、8 种 bowl；默认 2 plate + 1 bowl，也支持随机 2 个不同 plate + 1 bowl；每个 rollout 随机桌面 XY 和 yaw |
| `pick_paper_balls` | Office_10F_Room01 Area_7 + `pick_paper_balls_assets.yaml` 中的 3 种橘皮、3 种纸团、3 种桌面垃圾桶；支持 rollout 级物体种类、桌面 XY 和 yaw 泛化 |
| `pick_fruits_and_paper_balls` | Office_10F_Room01 组合桌面场景；优先使用 `Office_10F_Room01_fruits_paperballs.usd`，未提供时 fallback 到 `Office_10F_Room01.usd`。每个 rollout 包含 3 个 fruits + 1 个 bowl/plate + 1 个 orange peel + 1 个 paper ball + 1 个 trash can；当前支持 `default` / `random` 两种 rollout 泛化 |
| `pick_fruits_and_paper_balls_showroom` | 独立使用 Showroom `scene_1.usd` 的复杂桌面整理场景；固定显示 3 个水果 + 1 个碗 + 1 个垃圾桶 + 1 个纸团 + 1 个果皮；使用独立桌面坐标和机器人站位，当前不支持随机泛化。 |
| `pour_coffee` | Office_10F_Room01 Area_9 + `pour_coffee_assets.yaml` 中的 source container 和 target cups；使用 `Office_10F_Room01_pour_coffee.usd`，支持 rollout 级物体种类、桌面 XY/yaw 泛化，也支持 client 指定桌面 XY 的 custom 布局 |
| `pour_coffee_stereo` | `pour_coffee` 的双目 RGB 版本；左目 sensor key 为 `viewport_cam`，右目 sensor key 为 `viewport_cam_right`，图片尺寸为 640×480；继承倒咖啡 rollout 泛化协议 |
| `showroom_scene_2` | 开冰箱拿饮料并放置到厨房岛台；独立使用 `Showroom/scene_2/scene_2.usd`。当前提供固定布局的视觉与上肢 replay 控制链路。 |
| `showroom_scene_3` | 厨房岛台桌面清扫；独立使用 `Showroom/scene_3/scene_3.usd`。当前提供固定布局的视觉与上肢 replay 控制链路。 |
| `showroom_scene_4` | 脏衣服拾取、沙发整理、衣篓送洗衣机并装入；独立使用 `Showroom/scene_4/scene_4.usd`。当前 server 是固定基座 G1+O6，上肢 replay 可用，蹲下/行走/完整洗衣流程尚未接入。注意：该 USD 的 `washing_machine002` 子资产仍有外部 synthesis payload 引用，若需要完整洗衣机视觉，应先补齐/修正该资产引用。 |
| `laundryroom_r1_pro` | 一阶段 R1 Pro 洗衣场景；使用 `assets/environment/Laundryroom/Laundryroom_v1_0.usd`，受管机器人资产为 `assets/robot/r1_pro_usd/r1_pro.usd`。场景先加载并停用内嵌 `/Laundryroom/r1_pro`，再独立生成 `/Robot`。默认 root `(0.113757, 0.377053, 0.0)`，相对 USD 原始机器人站位保持前后方向不变、向机器人左侧移动约 0.815111m；默认站姿为 `standing=[0,0,0,0]`（直立、头部朝前）。支持 rollout 级 torso 和固定/随机 XY/yaw 站位，四路 640x480 RGB，必须使用 `r1pro32`。当前不执行底盘移动，也未接入洗衣成功评测。 |
| `showroom_scene_7` | 桌面切水果或黄瓜；独立使用 `Showroom/scene_7/scene_7.usd`。当前提供固定布局的视觉与上肢 replay 控制链路。 |
| `showroom_scene_8` | Showroom 桌面倒咖啡；独立使用 `Showroom/scene_8/scene_8.usd`，与 Room01 的 `pour_coffee` / `pour_coffee_stereo` 是两个独立 scene ID。当前未接入 Area_9 的咖啡对象泛化和 PBD 咖啡液评测。 |
| `showroom_scene_9` | 用遥控器打开电视；独立使用 `Showroom/scene_9/scene_9.usd`。当前提供固定布局的视觉与上肢 replay 控制链路。 |
| `showroom_scene_10` | 三阶魔方；独立使用 `Showroom/scene_10/Rubiks_Cube_physics.usdz`。资产包没有 `scene_10.usd`，当前用于资产加载和 replay 控制验证，魔方转层控制器/成功评测尚未接入。 |
| `showroom_scene_11` | 字母方块拼词；独立使用 `Showroom_standalone/scene_11/scene_11.usd`。默认机器人 root 为 `(0.003253, 1.395587, 0.76)`，与 `showroom_scene_11_wuji` 保持相同的机器人-桌子相对站位。支持 `single_custom`、`single_random`、`thirteen_custom`、`thirteen_random` 四种 rollout 模式；13 字母模式默认按 tmp 预生成数据泛化积木位置、yaw、积木颜色和桌面颜色，字体固定黑色且大小固定。每次 `reset_env`/`subscribe_step_result` 可用 `showroom_scene_11_robot_pose` 设置绝对位置 `pos` 或桌面相对位置 `table_relative_pos`（二选一），并用 `quat_wxyz` 设置 root 姿态；省略泛化字段时保持原 USD 的 26 字母布局。目标词/顺序自动评测尚未接入。 |
| `showroom_scene_11_wuji` | `showroom_scene_11` 的 G1+Wuji 版本；使用同一份 `Showroom_standalone/scene_11/scene_11.usd`、默认机器人 root `(0.003253, 1.395587, 0.76)` 和同一套四种字母泛化。必须使用 `action_layout=wuji54`；13 字母模式共享 tmp 预生成布局规则，不复用水果/纸团泛化，也没有自动拼词评测。机器人位置和姿态使用与 O6 版本完全相同的 `showroom_scene_11_robot_pose` 协议。 |

详细物体位姿、初始姿态见 [`SCENE_SWITCH_CLIENT_GUIDE.md`](SCENE_SWITCH_CLIENT_GUIDE.md) §2。

### 1.3 Action layout

| layout | dim | hand DoFs | 说明 |
|---|---|---|---|
| `full36` (代码默认 / legacy) | 36 | 22（每指 mcp + dip + 拇指 cmc/ip）| 14 个双臂关节和 22 个 O6 手关节由 action term 驱动 |
| `arm26` | 26 | 12（每指 mcp_pitch + 拇指 cmc_yaw/pitch） | 10 个 dip/ip 由 PD 撑住，arm26 模式下手指**不主动合拢**，适合策略只输出 mcp 层级时使用 |
| `wuji54` | 54 | 40（左右手各 20 DoF） | 14 个双臂关节 + 两只 Wuji 20-DoF 灵巧手；只允许用于 `pick_fruit_wuji`、`showroom_scene_11_wuji` 等 Wuji scene ID |
| `r1pro32` | 32 | 2 个二值夹爪命令 | R1 Pro 左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 保留 base 3 + padding 13；仅用于 `laundryroom_r1_pro`，索引 16–31 不执行 |

详见 [`ACTION_LAYOUT_GUIDE.md`](ACTION_LAYOUT_GUIDE.md)。

---

## 2. WS 协议消息一览

每条消息都是 JSON 文本，必含 `"type": "..."` 字段。

### 2.1 Client → Server

| `type` | 用途 | 关键字段 | 备注 |
|---|---|---|---|
| `ping` | 健康检查 | — | 立即返回 `pong` |
| `status` | 查询状态/能力 | — | 返回 scene_id / action_layout / action_dim 等 |
| `subscribe_step_result` | 订阅闭环帧推送 / 开始新 rollout | Scene11 可选 `showroom_scene_11_generalization`: dict / `showroom_scene_11_robot_pose`: dict；水果 scene 可选 `pick_fruit_active_fruits`: list[str]、`pick_fruit_single_fruit`: str、`pick_fruit_custom_initial_state`: dict、`pick_fruit_negative_sample_type`: str、`pick_fruit_robot_generalize`: bool；water_plant 可选 `water_plant_active_objects`: dict；stack_plate_bowl 可选 `stack_plate_bowl_generalization`: dict；pick_paper_balls 可选 `pick_paper_balls_generalization`: dict / `pick_paper_balls_active_objects`: dict\|list\|`"random"` / `pick_paper_balls_custom_initial_state`: dict；pick_fruits_and_paper_balls 可选 `pick_fruits_and_paper_balls_generalization`: dict；pour_coffee 可选 `pour_coffee_generalization`: dict / `pour_coffee_active_objects`: dict\|list\|`"random"`；R1 Pro torso 可选互斥的 `r1_pro_torso_preset`: str / `r1_pro_torso_pose`: 4D list，站位可选互斥的 `r1_pro_robot_pose`: dict / `r1_pro_robot_generalization`: dict\|bool | 各场景默认 reset，并在 rollout 边界应用各自支持的泛化；`pick_fruits_and_paper_balls` 当前只支持 `default` / `random`，`pick_paper_balls` 和倒咖啡场景支持 `default` / `random` / `fixed` / `custom`；`pick_fruit_robot_generalize=true` 只随机手臂/手指初始关节，不改 root 站位/朝向；R1 Pro torso/站位配置会强制 reset；Scene11 的两个专用字段只对 `showroom_scene_11` / `showroom_scene_11_wuji` 生效。 |
| `reset_env` | 显式开始新 rollout | Scene11 同样可带 `showroom_scene_11_generalization` / `showroom_scene_11_robot_pose`；pick_fruit 同样可带 `pick_fruit_active_fruits` / `pick_fruit_single_fruit` / `pick_fruit_custom_initial_state` / `pick_fruit_negative_sample_type` / `pick_fruit_generalize:false` / `pick_fruit_robot_generalize:true`；water_plant 同样可带 `water_plant_active_objects` / `water_plant_generalize:false`；stack_plate_bowl 同样可带 `stack_plate_bowl_generalization` / `stack_plate_bowl_generalize:false`；pick_paper_balls 同样可带 `pick_paper_balls_generalization` / `pick_paper_balls_generalize:false`；pick_fruits_and_paper_balls 同样可带 `pick_fruits_and_paper_balls_generalization` / `pick_fruits_and_paper_balls_generalize:false`；pour_coffee 同样可带 `pour_coffee_generalization` / `pour_coffee_generalize:false`；R1 Pro 可带 torso 字段及互斥的 `r1_pro_robot_pose` / `r1_pro_robot_generalization`；评测时可额外带 `evaluation_session_id` | 立即 reset；若当前连接已订阅 `step_result`，会额外主动推一条新的 `phase=initial`。同一请求中可同时设置 Scene11 字母泛化和机器人位姿。 |
| `start_evaluation` | 创建 server 端评测 session | `scene_id`、`evaluation.task`、`evaluation.metrics`、可选 `evaluation.debug_details` | `objects_in_container` 当前支持 5 个水果 scene ID（`pick_fruit` / `pick_fruit_wuji` / `pick_fruit_office1` / `pick_fruit_lidar` / `pick_fruit_stereo`）和 `pick_paper_balls` |
| `finish_episode` | 结束当前 rollout 并生成单次评测 report | `evaluation_session_id` | 需在动作执行结束且队列为空后调用 |
| `finish_evaluation` | 结束评测 session 并获取多 rollout 汇总 | `evaluation_session_id` | server 端计算通过率，client 不需要自己汇总 |
| `start_pose_debug` | 调试：记录 `torso_link` / viewport camera pose | `duration_s`: float，默认 10；可选 `label`: str | 只用于排查视角/机身抖动；server 在后续 `env.step` 时写 JSONL 和 summary |
| `stop_pose_debug` | 调试：提前结束 pose 记录 | — | 返回最近一次 summary；别名 `finish_pose_debug` |
| `unsubscribe_step_result` | 取消订阅 | — | |
| `request_observation` | 一次性拉观测 | `keys`: list[str] (可选) | REST 风格；不订阅时用 |
| `submit_actions` | 推一段动作 | `actions`: list[list[float]] | 内层长度 = action_dim |
| `set_control_mode` | 切换运行期控制模式 | `control_mode`: `"passthrough"`\|`"novus_consistency"` | 不触发重启，只能在队列空闲时切换；默认 `passthrough`，一致性验证/仿真-真机延迟对齐时显式切 `novus_consistency` |
| `switch_scene` | 切场景（独占，触发自重启）| `scene_id`: str | 见 [`SCENE_SWITCH_CLIENT_GUIDE.md`](SCENE_SWITCH_CLIENT_GUIDE.md) |
| `switch_action_layout` | 切 layout（独占，触发自重启）| `action_layout`: str | 见 [`ACTION_LAYOUT_GUIDE.md`](ACTION_LAYOUT_GUIDE.md) §3.4 |
| `switch_scene_and_layout` | 原子切 scene + layout（独占，触发一次自重启） | `scene_id`: str、`action_layout`: str | 跨 O6/Wuji/R1 Pro 必须使用；成功 ack 后重连并确认 `status` 的 scene/layout/dim |
| `switch_pick_fruit_layout` | pick_fruit 桌面 3 选 leaves（独占，触发自重启） | `active_fruits`: list[str]\|`"random"`，可选 `seed`: int | 用于 `pick_fruit` / `pick_fruit_wuji` / `pick_fruit_lidar` / `pick_fruit_stereo`，不用于 `pick_fruit_office1`；它只改水果启动期 layout，不改机器人或 action layout；见 [`SCENE_SWITCH_CLIENT_GUIDE.md`](SCENE_SWITCH_CLIENT_GUIDE.md) §8 |

### 2.2 Server → Client

| `type` | 时机 | 关键字段 |
|---|---|---|
| `pong` | 应答 ping | `ok` / `timestamp` |
| `status_response` | 应答 status | `scene_id` / `robot` / `action_dim` / `action_layout` / `control_mode` / `valid_control_modes` / `unactuated_joint_names` / `controlled_action_indices` / `ignored_action_indices` / `viewport_cam_key` / `multi_camera_enabled` / `image_camera_keys` / `camera_calibration` / `stereo_enabled` / `stereo_camera_keys` / `stereo_calibration` / `r1_pro_torso` / `r1_pro_robot_pose` / `pick_fruit_*` / `water_plant_*` / `stack_plate_bowl_*` / `pick_paper_balls_*` / `pick_fruits_and_paper_balls_*` / `pour_coffee_*` / `showroom_scene_11_generalization_mode` / `showroom_scene_11_active_letters` / `showroom_scene_11_layout` / `showroom_scene_11_physx_synced_bodies` / `showroom_scene_11_robot_pose` / `pose_debug_active` / `pose_debug_last_summary` / `ws_client_count` ...。Scene11 尚未 reset 时，相关 rollout 回显字段可能为 `null` 或 `0`。 |
| `subscribe_step_result_response` | 应答 subscribe | `ok` / `subscribed` / `action_dim` / `action_layout` / `rollout_reset_applied`；R1 Pro 返回 `r1_pro_torso` 与 `r1_pro_robot_pose` target/PhysX feedback；pick_fruit reset 后返回 `pick_fruit_*`；stack_plate_bowl reset 后返回 `stack_plate_bowl_*`；pick_paper_balls reset 后返回 `pick_paper_balls_*`；pour_coffee reset 后返回 `pour_coffee_*`；Scene11 reset 后返回 `showroom_scene_11_generalization_mode` / `showroom_scene_11_active_letters` / `showroom_scene_11_layout` / `showroom_scene_11_physx_synced_bodies` / `showroom_scene_11_robot_pose`，其中 `robot_pose` 同时包含目标值和实际 PhysX root 位姿。 |
| `reset_env_response` | 应答 reset_env | `ok` / `initial_pushed` / `step_result_subscribed`；R1 Pro 返回 `r1_pro_torso` 与 `r1_pro_robot_pose` target/PhysX feedback；水果、盘子碗、纸团/果皮和倒咖啡场景返回各自的 active/layout 字段；Scene11 reset 后返回 `showroom_scene_11_generalization_mode` / `showroom_scene_11_active_letters` / `showroom_scene_11_layout` / `showroom_scene_11_physx_synced_bodies` / `showroom_scene_11_robot_pose`。 |
| `set_control_mode_response` | 应答 set_control_mode | `ok` / `control_mode` / `previous_control_mode` / `filter_reset` / `valid_control_modes`；忙时会返回 `ok=false` 和当前队列状态 |
| `start_evaluation_response` | 应答 start_evaluation | `ok` / `evaluation_session_id` / `scene_id` / `task` / `metrics` |
| `evaluation_report` | 应答 finish_episode | `ok` / `mode` / `active_objects` / `metrics`；`debug_details` 由 start_evaluation 控制 |
| `evaluation_summary` | 应答 finish_evaluation | `ok` / `total_rollouts` / `pass_rates` / `in_container_count_distribution` / `combo_coverage` / `pass_rate_by_combo` |
| `start_pose_debug_response` | 应答 start_pose_debug | `ok` / `active` / `path` / `summary_path` / `duration_s` |
| `pose_debug_summary` | 应答 stop_pose_debug，或 status 中的最近结果 | `ok` / `reason` / `path` / `summary_path` / `sample_count` / `stats` |
| `unsubscribe_step_result_response` | 应答 unsubscribe | `ok` / `subscribed` |
| `observation_response` | 应答 request_observation | `obs`: dict[str, ndarray-payload] |
| `submit_actions_response` | 应答 submit_actions | `ok` / `accepted_count` / `target_executed` |
| `step_result` | **主动推送**：初始帧 + 每批动作执行完 | `phase`("initial"/"post_actions") / `frames` / `action_layout`；Scene11 额外在消息顶层返回 `showroom_scene_11_generalization_mode` / `showroom_scene_11_active_letters` / `showroom_scene_11_layout` / `showroom_scene_11_physx_synced_bodies` / `showroom_scene_11_robot_pose`；双目场景额外返回 `stereo_*`；R1 Pro 返回 `observation.images` 四路图像、`camera_calibration`、`r1_pro_torso`、`r1_pro_robot_pose` 和模型状态字段。 |
| `switch_scene_response` | 应答 switch_scene | `ok` / `scene_id` |
| `scene_switching` | switch_scene 后异步推送 | `scene_id` (server 即将 exit) |
| `switch_scene_and_layout_response` | 应答原子切换 | `ok` / `scene_id` / `action_layout` / `from_scene_id` / `from_action_layout` / `restart_required` |
| `scene_and_layout_switching` | 原子切换状态文件写成功后推送 | `scene_id` / `action_layout` (server 即将 exit) |
| `switch_action_layout_response` | 应答 switch_action_layout | `ok` / `action_layout` / `from_action_layout` |
| `action_layout_switching` | switch_action_layout 后异步推送 | `action_layout` (server 即将 exit) |
| `switch_pick_fruit_layout_response` | 应答 switch_pick_fruit_layout | `ok` / `active_fruits` / `from_active_fruits` |
| `pick_fruit_layout_switching` | switch_pick_fruit_layout 后异步推送 | `active_fruits` (server 即将 exit) |

`step_result.frames[i]` 字段：

```jsonc
{
  "image": {"shape":[H,W,3], "encoding":"jpeg_base64"|"png_base64"|"raw_base64", "data":"..."},
  "images": {                         // 仅双目场景；image 仍等价于 images.left
    "left":  {"shape":[H,W,3], "encoding":"jpeg_base64"|"png_base64"|"raw_base64", "data":"..."},
    "right": {"shape":[H,W,3], "encoding":"jpeg_base64"|"png_base64"|"raw_base64", "data":"..."}
  },
  "state": {
    "joint_position": [...],            // 通用字段：O6 51 维，Wuji 69 维，R1 Pro 22 维
    "joint_velocity":  [...],
    "joint_position_51": [...],         // 仅 O6；Wuji 对应 joint_position_69
    "joint_velocity_51":  [...],        // 仅 O6；Wuji 对应 joint_velocity_69
    "hand_joint_state": [...],          // 通用字段：O6 22 维，Wuji 40 维
    "hand_joint_state_22": [...],       // 仅 O6；Wuji 对应 hand_joint_state_40
    "right_eef_pos":  [x,y,z],
    "right_eef_quat": [w,x,y,z],
    "left_eef_pos":   [x,y,z],
    "left_eef_quat":  [w,x,y,z],
    "object_root_pose_7":     [...],    // 主操作物 (scene 决定)
    "object_root_velocity_6": [...],
    "object2_root_pose_7":    [...],    // 仅 cereal / pick_fruit 等多物场景
    ...
  },
  "external_action": [...],             // 推荐读取；实际长度为 26、36、54 或 32
  "applied_action": [...],              // 推荐读取；实际送入 env/PhysX 的 action
  "external_action_36": [...],          // 兼容旧字段名，实际长度仍为 26、36 或 54
  "applied_action_36": [...],           // 兼容旧字段名，实际长度仍为 26、36 或 54
  "external_action_layout": "full36"|"arm26"|"wuji54"|"r1pro32",
  "control_mode": "passthrough"|"novus_consistency",
  "executed_actions": <int>,
  "wall_time": <float>
}
```

双目场景 `pick_fruit_stereo` / `pour_coffee_stereo` 的 `step_result.observation.images` 和 `step_result.frames[i].images` 都按 `left/right` 返回图片；旧字段 `image` 保持为左目，方便老 client 继续运行。`status_response.stereo_camera_keys` 当前为 `{"left": "viewport_cam", "right": "viewport_cam_right"}`，`stereo_calibration` 原样携带当前 K/D/R/T/R1/R2/P1/P2/Q，以及左目 `T_world2cam`；其中 `T_world2cam` 的 world 坐标按 `pelvis` 解释，并参与当前 `head_link` 双目渲染挂载。

### 2.3 state 51/69/22 维与 action 26/36/54/32 维说明

Client 侧不要 hardcode 维度，连接后先发 `{"type":"status"}`，以返回值为准：

| 字段 | 含义 |
|---|---|
| `action_dim` | `submit_actions.actions[*]` 必须使用的长度：`26`、`36`、`54` 或 `32` |
| `action_layout` | 当前 action layout：`arm26`、`full36`、`wuji54` 或 `r1pro32` |
| `action_joint_names` | 当前 action 向量每一维对应的 joint 名称 |
| `unactuated_joint_names` | 当前 layout 不由 action 驱动的 joint；`full36=[]`，`arm26` 为 10 个 DIP/IP |
| `step_result_state_dim` / `step_result_state_layout` | O6 为 `51 / full_body_51`；Wuji 为 `69 / full_body_69`；R1 Pro 为 `22 / full_body_22` |
| `full_body_joint_names` | `joint_position_<dim>` / `joint_velocity_<dim>` 每一维对应的 joint 名称 |

O6 场景的 `step_result.frames[*].state.joint_position_51` 与 `joint_velocity_51` 是同一套 51 维全身顺序，当前环境的索引如下：

```text
0  left_hip_pitch_joint
1  right_hip_pitch_joint
2  waist_yaw_joint
3  left_hip_roll_joint
4  right_hip_roll_joint
5  waist_roll_joint
6  left_hip_yaw_joint
7  right_hip_yaw_joint
8  waist_pitch_joint
9  left_knee_joint
10 right_knee_joint
11 left_shoulder_pitch_joint
12 right_shoulder_pitch_joint
13 left_ankle_pitch_joint
14 right_ankle_pitch_joint
15 left_shoulder_roll_joint
16 right_shoulder_roll_joint
17 left_ankle_roll_joint
18 right_ankle_roll_joint
19 left_shoulder_yaw_joint
20 right_shoulder_yaw_joint
21 left_elbow_joint
22 right_elbow_joint
23 left_wrist_roll_joint
24 right_wrist_roll_joint
25 left_wrist_pitch_joint
26 right_wrist_pitch_joint
27 left_wrist_yaw_joint
28 right_wrist_yaw_joint
29 lh_index_mcp_pitch
30 lh_middle_mcp_pitch
31 lh_pinky_mcp_pitch
32 lh_ring_mcp_pitch
33 lh_thumb_cmc_yaw
34 rh_index_mcp_pitch
35 rh_middle_mcp_pitch
36 rh_pinky_mcp_pitch
37 rh_ring_mcp_pitch
38 rh_thumb_cmc_yaw
39 lh_index_dip
40 lh_middle_dip
41 lh_pinky_dip
42 lh_ring_dip
43 lh_thumb_cmc_pitch
44 rh_index_dip
45 rh_middle_dip
46 rh_pinky_dip
47 rh_ring_dip
48 rh_thumb_cmc_pitch
49 lh_thumb_ip
50 rh_thumb_ip
```

Action 的顺序不是 `joint_position_51` 的子数组原顺序；它是 server action term 的顺序。

`full36` action：14 个双臂关节 + 22 个 O6 双手关节，长度 36。`arm26` action：同样的 14 个双臂关节 + 12 个 MCP/CMC 手部关节，长度 26；10 个 DIP/IP 不出现在 action 里，由 PD 维持。`wuji54` action：14 个双臂关节 + 左右 Wuji 手各 20 个关节，长度 54，对应 69 维全身状态和 40 维 `hand_joint_state_40`。`r1pro32` 是 14 个双臂绝对角 + 2 个二值夹爪 + 16 个保留槽位；它不是 32 个物理关节的一一映射。完整顺序见 [`ACTION_LAYOUT_GUIDE.md`](ACTION_LAYOUT_GUIDE.md) 和 `status_response.action_joint_names`。

当前 O6、Wuji 和一阶段 R1 Pro articulation 都使用 `fix_root_link=True`。G1 action layout 只驱动双臂/双手；R1 Pro `r1pro32` 只驱动双臂/夹爪，torso 与固定基座的 XY/yaw 站位由 rollout/reset 参数独立配置并在 rollout 内保持，底盘 3 维不执行。51/69/22 维 state 是观测，不代表所有维度都能由 action 控制。

提交动作时：

```python
status = ...  # status_response
action_dim = int(status["action_dim"])
action_names = status["action_joint_names"]
full_state_names = status["full_body_joint_names"]

# 每一行长度必须等于 action_dim。
await ws.send(json.dumps({
    "type": "submit_actions",
    "actions": [[0.0] * action_dim for _ in range(30)],
}))

# 如果要把 action joint 对应到 51 维 state，可按名字查索引。
action_state_indices = [full_state_names.index(n) for n in action_names]
```

---

## 3. CLI / 脚本工具速查

所有路径相对仓库根 `/root/code/easim/`。

| 用途 | 命令 |
|---|---|
| **主 client（rollout 模型）** | 在 client 仓库运行对应模型脚本，URI 指向 `ws://172.31.0.226:<port>`；本 server 工作区不包含模型 client |
| **手动切场景**（standalone） | （见 [`SCENE_SWITCH_CLIENT_GUIDE.md`](SCENE_SWITCH_CLIENT_GUIDE.md) §4）|
| **手动切 action_layout** | 直接发 WS：`{"type":"switch_action_layout","action_layout":"arm26"}`；或在脚本里 `sys.path.insert(0, "source/easim/scripts")` 后调用 `g1o6_action_layout.request_action_layout_switch` |
| **手动切 pick_fruit 启动期 layout** | `python source/easim/scripts/_switch_pick_fruit_layout.py --uri ws://172.31.0.226:8083 --active_fruits qsim_carrot,qsim_pumpkins,qsim_apple_2` 或 `--random [--seed N]`；通常 rollout 级泛化直接用 `pick_fruit_active_fruits`，详见 §4.1 |
| **整池重启** | 见 [`RESTART_GUIDE.md`](RESTART_GUIDE.md) §1.1（带 `ACTION_LAYOUT=` env） |
| **协议层 smoke**（pool 4 worker）| `python3 _smoke_test_pool.py` |
| **H5 open-loop 回放** | `python source/easim/scripts/_h5_replay.py --url ws://172.31.0.226:8083 --h5 <path> --key <dataset_key>` |
| **仿真-真机控制延迟分析** | `python tools/simreal_control_lag_analysis.py --uri ws://127.0.0.1:8080 --out-dir /tmp/simreal_control_lag_analysis --summary-json /tmp/simreal_control_lag_analysis/summary.json`；会回放两份默认 Novus H5、计算 lag、输出 arm14 曲线图，结束后恢复 `passthrough` |
| **初始状态探测** | `python3 _probe_initial_state.py --uri ws://172.31.0.226:8080` |

主 client `client_demo_wam_policy_G1O6.py` 关键 flag：

```bash
--uri ws://172.31.0.226:<port>
--action_layout {auto,arm26,full36}        # auto = 从 status 读取（默认）
--switch_action_layout                     # 与 server 不一致时自动切（mirrors --switch_scene）
--switch_scene                             # 与 --task_name 配合，连接前先切场景
--task_name <key>                          # 见 TASK_NAME_TO_PROMPT / TASK_NAME_TO_SCENE_ID 映射
--actions_per_submit 30                    # batch 大小
--max_loops 3                              # 推理迭代数
--rollout_repeat_times 20                  # 同 worker 上重复跑多少次
--obs_log_dir <path>                       # 落盘观测帧（dense_video）
--save_dense_frames                        # 每帧 JPEG 一并落盘
--debug_h5_path <path> --debug_parquet_replay_ws  # 不加载模型，纯 H5 重放
```

完整 flag 列表 `--help`。

---

## 4. 端到端最简示例（Python）

下面是一个**完整的 closed-loop rollout 客户端**最小实现，假设 cfg 26-D，连 8083 worker：

```python
import asyncio, base64, json
import numpy as np
import websockets

URI = "ws://172.31.0.226:8083"
TARGET_LAYOUT = "arm26"   # 或 "full36"


async def ensure_layout(uri: str, target: str) -> dict:
    """如果 worker 当前 layout != target，发 switch_action_layout 并等重启。返回最终 status。"""
    import sys
    sys.path.insert(0, "source/easim/scripts")
    from g1o6_action_layout import request_action_layout_switch  # 共享 helper

    async with websockets.connect(uri, open_timeout=10) as ws:
        await ws.send(json.dumps({"type": "status"}))
        s = json.loads(await ws.recv())
    if s.get("action_layout") == target:
        return s
    return await request_action_layout_switch(uri, target,
                                              restart_timeout=240, poll_interval=5,
                                              initial_grace=5)


def policy(rgb: np.ndarray, qpos51: np.ndarray, layout_dim: int) -> np.ndarray:
    """你的策略：返回 (T, layout_dim) action。这里用零向量占位。"""
    return np.zeros((30, layout_dim), dtype=np.float32)


async def run_rollout():
    s = await ensure_layout(URI, TARGET_LAYOUT)
    layout_dim = int(s["action_dim"])
    print(f"[rollout] worker scene={s['scene_id']} layout={s['action_layout']} dim={layout_dim}")

    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        # 订阅 + 拿 t=0 帧
        await ws.send(json.dumps({"type": "subscribe_step_result"}))
        ack = json.loads(await ws.recv())
        init = json.loads(await ws.recv())
        assert init["phase"] == "initial"

        # 解码工具
        def decode_image(p):
            raw = base64.b64decode(p["data"])
            arr = np.frombuffer(raw, dtype=np.uint8)
            if p["encoding"] in ("jpeg_base64", "png_base64"):
                import cv2
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return arr.reshape(p["shape"])  # raw

        frame = init["frames"][0]
        rgb = decode_image(frame["image"])
        qpos = np.array(frame["state"]["joint_position_51"], dtype=np.float32).reshape(-1)

        # 闭环
        for step in range(5):
            actions = policy(rgb, qpos, layout_dim).tolist()
            await ws.send(json.dumps({"type": "submit_actions", "actions": actions}))
            ack2 = json.loads(await ws.recv())  # submit_actions_response
            assert ack2["ok"], ack2
            sr = json.loads(await ws.recv())   # post_actions step_result
            assert sr["phase"] == "post_actions"
            last = sr["frames"][-1]
            rgb = decode_image(last["image"])
            qpos = np.array(last["state"]["joint_position_51"], dtype=np.float32).reshape(-1)
            print(f"  step {step} done; executed_actions={sr.get('executed_actions')}")


asyncio.run(run_rollout())
```

### 4.1 pick_fruit 泛化模式示例

水果池由 `source/easim/scenario_loader/configs/pick_fruits_assets.yaml` 配置，目前包含 100 个 fruit leaf（不含 `qsim_red_bowl`）。运行时只会从当前 USD stage 能解析到的 fruit prim 里采样；如果固定指定了当前资产缺失的 leaf，server 会返回 `ok=false`，错误里会说明 requested fruit pool 在当前 USD 不可用。

本节的固定三水果、随机三水果、单水果、负样例、自定义初始状态和 PhysX 评测请求同时适用于 `pick_fruit_wuji`。区别只有机器人和 action/state contract：`pick_fruit` 使用 G1+O6 的 `arm26/full36`，`pick_fruit_wuji` 必须使用 `wuji54`。

#### `pick_fruit_wuji` 完整启动与 rollout 示例

跨 O6/Wuji 机器人时，scene 和 layout 必须在同一次冷启动中配对。client 不要依次发送 `switch_action_layout` 和 `switch_scene`，应使用原子协议：

```json
{
  "type": "switch_scene_and_layout",
  "scene_id": "pick_fruit_wuji",
  "action_layout": "wuji54"
}
```

推荐直接运行 helper；它会等待 worker 重启，并验证 `scene_id`、`action_layout`、`action_dim` 三项：

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  source/easim/scripts/_switch_scene_and_layout.py \
  --uri ws://172.31.0.226:8081 \
  --scene_id pick_fruit_wuji \
  --action_layout wuji54
```

切回普通 O6 水果场景：

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  source/easim/scripts/_switch_scene_and_layout.py \
  --uri ws://172.31.0.226:8081 \
  --scene_id pick_fruit \
  --action_layout arm26
```

下面的手工启动方式仍保留用于首次部署或状态文件故障恢复：

```bash
PORT=8081 \
INITIAL_SCENE=pick_fruit_wuji \
ACTION_LAYOUT=wuji54 \
SCENE_STATE_FILE=/tmp/pick_fruit_wuji_scene_8081 \
ACTION_LAYOUT_STATE_FILE=/tmp/pick_fruit_wuji_layout_8081 \
PICK_FRUIT_LAYOUT_STATE_FILE=/tmp/pick_fruit_wuji_fruits_8081 \
  ./run_g1o6_server_loop.sh
```

`pick_fruit_wuji` 支持的 rollout 模式：

| 模式 | rollout 请求字段 |
|---|---|
| 随机三水果 | 不传 `pick_fruit_active_fruits` / `pick_fruit_single_fruit` |
| 固定三水果 | `"pick_fruit_active_fruits": [leaf1, leaf2, leaf3]` |
| 单水果 | `"pick_fruit_single_fruit": "apple"`，或传单元素 `pick_fruit_active_fruits` |
| 负样例 | `"pick_fruit_negative_sample_type": "fruits_in_bowl"` / `"no_fruit"` / `"empty_table"` |
| 自定义初始状态 | `"pick_fruit_custom_initial_state": {...}`，格式与本节 4.1.1 相同 |
| 机器人姿态泛化 | `"pick_fruit_robot_generalize": true`；随机 14 个双臂 + 40 个 Wuji 手关节，root 位置/yaw 不变 |

下面的完整 client 示例启动一个单苹果 rollout，开启 Wuji 关节泛化，保存初始图片，动态从 69D state 生成 54D 保持动作，再保存动作后图片：

```python
import asyncio
import base64
import json
from pathlib import Path

import websockets

URI = "ws://172.31.0.226:8081"
OUT = Path("pick_fruit_wuji_rollout")
OUT.mkdir(exist_ok=True)


async def recv_type(ws, expected, timeout=180):
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") == "error":
            raise RuntimeError(msg)
        if msg.get("type") in expected:
            return msg


def save_frame(frame, path):
    path.write_bytes(base64.b64decode(frame["image"]["data"]))


async def main():
    async with websockets.connect(
        URI,
        open_timeout=20,
        max_size=64 * 1024 * 1024,
        ping_interval=None,
    ) as ws:
        await ws.send(json.dumps({"type": "status"}))
        status = await recv_type(ws, {"status_response"}, 20)
        assert status["scene_id"] == "pick_fruit_wuji", status
        assert status["robot"] == "unitree_g1_wuji_bimanual_jointpos", status
        assert status["action_layout"] == "wuji54", status
        assert status["action_dim"] == 54, status
        assert status["step_result_state_dim"] == 69, status

        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
            "pick_fruit_single_fruit": "apple",
            "pick_fruit_robot_generalize": True,
        }))
        ack = await recv_type(ws, {"subscribe_step_result_response"})
        assert ack["ok"], ack
        assert ack["pick_fruit_active_fruits"] == ["qsim_apple_2"], ack
        assert ack["pick_fruit_robot_pose"]["root_pose_randomized"] is False
        assert ack["pick_fruit_robot_pose"]["matched_joints"] == 54

        initial = await recv_type(ws, {"step_result"})
        assert initial["phase"] == "initial", initial
        initial_frame = initial["frames"][-1]
        assert len(initial_frame["state"]["hand_joint_state_40"]) == 40
        save_frame(initial_frame, OUT / "initial.jpg")

        qpos69 = initial_frame["state"]["joint_position"]
        full_index = {
            name: index
            for index, name in enumerate(status["full_body_joint_names"])
        }
        action_names = (
            status.get("action_resolved_joint_names")
            or status["action_joint_names"]
        )
        hold_action = [float(qpos69[full_index[name]]) for name in action_names]
        assert len(hold_action) == 54

        await ws.send(json.dumps({
            "type": "submit_actions",
            "actions": [hold_action] * 3,
        }))
        submit_ack = await recv_type(ws, {"submit_actions_response"})
        assert submit_ack["ok"], submit_ack
        post_actions = await recv_type(ws, {"step_result"})
        assert post_actions["phase"] == "post_actions", post_actions
        save_frame(post_actions["frames"][-1], OUT / "post_actions.jpg")


asyncio.run(main())
```

水果评测也可直接把 `start_evaluation.scene_id` 改为 `pick_fruit_wuji`。评测仍读取水果和碗的运行时 PhysX world pose，与机器人手型和 action layout 无关。

#### `pick_fruit_office1` 独立场景

`pick_fruit` 固定加载 `Office_10F_Room01_pick_fruits.usd`；`pick_fruit_office1` 固定加载 `office_1.usd`，两者不再通过 `EASIM_PICK_FRUITS_ROOM01_USD` 互相覆盖。office1 当前只解析到 `qsim_pumpkins` 和 `qsim_carrot` 两个受管水果，建议显式传单元素 `pick_fruit_active_fruits`；三水果 `switch_pick_fruit_layout` 不适用于该场景。

下面示例完成场景切换、等待 worker 重启，并启动一个胡萝卜 rollout：

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8083"


async def main():
    async with websockets.connect(URI, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "switch_scene",
            "scene_id": "pick_fruit_office1",
        }))
        ack = json.loads(await ws.recv())
        assert ack["ok"] and ack["scene_id"] == "pick_fruit_office1"

    for _ in range(120):
        await asyncio.sleep(2)
        try:
            async with websockets.connect(URI, open_timeout=3, max_size=None) as ws:
                await ws.send(json.dumps({"type": "status"}))
                status = json.loads(await ws.recv())
                if status.get("ok") and status.get("scene_id") == "pick_fruit_office1":
                    break
        except Exception:
            continue
    else:
        raise TimeoutError("pick_fruit_office1 worker restart timed out")

    async with websockets.connect(URI, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
            "pick_fruit_active_fruits": ["qsim_carrot"],
        }))
        subscribe_ack = json.loads(await ws.recv())
        initial = json.loads(await ws.recv())
        assert subscribe_ack["ok"]
        assert subscribe_ack["pick_fruit_active_fruits"] == ["qsim_carrot"]
        assert initial["type"] == "step_result" and initial["phase"] == "initial"


asyncio.run(main())
```

##### Office1 桌子、机器人位置和“向后退”

`pick_fruit_office1` 的桌子属于 `office_1.usd` 环境资产，位置和朝向由 USD 固定。当前 rollout
协议**没有** `table` / `table_xyz` / `table_offset` 字段，client 不能在运行时平移整张桌子：

- `pick_fruit_custom_initial_state.bowl.xy` 和 `fruits.<leaf>.xy` 设置桌面局部 XY；
- `pick_fruit_custom_initial_state.robot.pos` 设置机器人 root 的绝对 env-local `[x, y, z]`；
- `robot.yaw_z_deg` 设置机器人 env-local yaw；
- 如需真正移动桌子，应制作独立 room USD/scene，避免修改 `office_1.usd` 后影响同一资产的其它用途。

当前 office1 默认机器人位姿为：

```python
DEFAULT_ROBOT_POS = [-0.2513279542, 2.7930426597, 0.76]
DEFAULT_ROBOT_YAW_Z_DEG = 30.0
```

在这个朝向下，机器人局部 `+X` 指向桌子。“向后退 `d` 米”需要沿机器人局部 `-X`
移动；换算到 env-local XY 为：

```python
import math


def office1_robot_back_pos(distance_m: float) -> list[float]:
    x0, y0, z0 = -0.2513279542, 2.7930426597, 0.76
    yaw = math.radians(30.0)
    return [
        x0 - distance_m * math.cos(yaw),
        y0 - distance_m * math.sin(yaw),
        z0,
    ]
```

| 相对默认站位 | `robot.pos` |
|---|---|
| 不后退 | `[-0.2513279542, 2.7930426597, 0.76]` |
| 后退 0.5m | `[-0.6843406561, 2.5430426597, 0.76]` |
| 后退 1.0m | `[-1.1173533580, 2.2930426597, 0.76]` |

下面是固定胡萝卜、设置桌面物体，并让机器人后退 0.5m 的完整 rollout 请求。需要后退
1m 时只把 `BACK_DISTANCE_M` 改为 `1.0`：

```python
import json


BACK_DISTANCE_M = 0.5
custom_state = {
    "custom_policy": "per_rollout",
    "active_fruits": ["qsim_carrot"],
    "bowl": {
        "xy": [0.53, 0.15],
        "yaw_z_deg": 0.0,
    },
    "fruits": {
        "qsim_carrot": {
            "xy": [0.20, 0.08],
            "yaw_z_deg": 30.0,
        },
    },
    "robot": {
        "pos": office1_robot_back_pos(BACK_DISTANCE_M),
        "yaw_z_deg": 30.0,
    },
}

await ws.send(json.dumps({
    "type": "subscribe_step_result",
    "force_reset": True,
    "pick_fruit_custom_initial_state": custom_state,
}))
ack = json.loads(await ws.recv())
initial = json.loads(await ws.recv())
assert ack["ok"], ack
actual_pos = ack["pick_fruit_robot_pose"]["pos"]
expected_pos = custom_state["robot"]["pos"]
assert all(abs(a - b) < 1e-5 for a, b in zip(actual_pos, expected_pos))
assert initial["type"] == "step_result" and initial["phase"] == "initial"
```

`custom_policy="per_rollout"` 表示每个 rollout 都由 client 发送本轮完整布局。若下一轮仍需相同的
机器人、碗和水果位置，应再次发送同一份 `pick_fruit_custom_initial_state`。


机器人初始关节泛化是独立开关，默认关闭。需要让 rollout 之间随机机器人手臂/手指初始关节时，给 `subscribe_step_result` 或 `reset_env` 增加 `"pick_fruit_robot_generalize": true`；也可以写在嵌套对象里：`"pick_fruit_generalization": {"robot_generalize": true}`。这里对齐 `easim_ck` 的 teleop/reset 路径：只随机手臂和手指关节，不随机 root 位置/yaw。O6 `pick_fruit` 会匹配 24 个关节（14 手臂 + 10 O6 手指）；`pick_fruit_wuji` 会匹配 54 个关节（14 手臂 + 40 Wuji 手指）。普通 `pick_fruit` 保持其默认站位 `(0.35, 1.80, 0.76)`；`pick_fruit_office1` 保持上文的 office1 默认站位，或同一份 custom initial state 中显式指定的 root 位姿。reset ack 中 `pick_fruit_robot_pose.root_pose_randomized=false` 且 `matched_joints=24` 或 `54` 时表示本次已生效；`pick_fruit_robot_generalize` 回显本次是否开启。

只想泛化机器人手臂/手指初始关节、不改水果/碗组合和位置时，可以这样开新 rollout：

```python
await ws.send(json.dumps({
    "type": "reset_env",
    "pick_fruit_generalize": False,
    "pick_fruit_robot_generalize": True,
}))
ack = json.loads(await ws.recv())
init = json.loads(await ws.recv())
assert ack["pick_fruit_generalization_mode"] == "robot_joints_only"
assert ack["pick_fruit_robot_pose"] is not None
```

模式 1：client 指定三种水果；后续 rollout 固定这三种水果，只随机位置和 pose。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8083"
FIXED_FRUITS = ["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"]


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "pick_fruit_active_fruits": FIXED_FRUITS,
            # 可选：同时泛化机器人手臂/手指初始关节；不改 root 站位/朝向。
            "pick_fruit_robot_generalize": True,
        }))

        ack = json.loads(await ws.recv())   # subscribe_step_result_response
        init = json.loads(await ws.recv())  # phase=initial step_result

        assert ack["ok"], ack
        assert ack["pick_fruit_generalization_mode"] == "fixed_fruits"
        assert ack["pick_fruit_active_fruits"] == FIXED_FRUITS
        assert init["phase"] == "initial"

        # reset 后，server 会返回三种水果 + 碗从左到右的顺序。
        # 例：['胡萝卜', '苹果', '碗', '南瓜']
        assert len(ack["pick_fruit_left_to_right_names"]) == 4
        assert "碗" in ack["pick_fruit_left_to_right_names"]
        print("fixed rollout fruits:", ack["pick_fruit_active_fruits"])
        print("fixed left-to-right:", ack["pick_fruit_left_to_right_names"])


asyncio.run(main())
```

如果同一条连接已经订阅过，只想显式开始下一次 rollout，也可以在同一个 async 函数里发 `reset_env`，字段完全一样：

```python
await ws.send(json.dumps({
    "type": "reset_env",
    "pick_fruit_active_fruits": FIXED_FRUITS,
    "pick_fruit_robot_generalize": True,  # 可选
}))
ack = json.loads(await ws.recv())   # reset_env_response
init = json.loads(await ws.recv())  # 如果当前连接已订阅，会继续推 phase=initial
```

模式 2：client 不指定水果；server 每次 rollout 从可解析水果池里自动选 3 个，并随机位置和 pose。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8083"


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "pick_fruit_robot_generalize": True,  # 可选
        }))

        ack = json.loads(await ws.recv())   # subscribe_step_result_response
        init = json.loads(await ws.recv())  # phase=initial step_result

        assert ack["ok"], ack
        assert ack["pick_fruit_generalization_mode"] == "random_fruits"
        assert len(ack["pick_fruit_active_fruits"]) == 3
        assert init["phase"] == "initial"
        assert len(ack["pick_fruit_left_to_right_names"]) == 4
        assert "碗" in ack["pick_fruit_left_to_right_names"]
        print("random rollout fruits:", ack["pick_fruit_active_fruits"])
        print("random left-to-right:", ack["pick_fruit_left_to_right_names"])


asyncio.run(main())
```

模式 3：client 指定单个水果；后续 rollout 只激活这个水果和碗，并随机该水果与碗的位置和 pose。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8083"


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "pick_fruit_single_fruit": "apple",
        }))

        ack = json.loads(await ws.recv())   # subscribe_step_result_response
        init = json.loads(await ws.recv())  # phase=initial step_result

        assert ack["ok"], ack
        assert ack["pick_fruit_generalization_mode"] == "single_fruit"
        assert ack["pick_fruit_active_fruits"] == ["qsim_apple_2"]
        assert init["phase"] == "initial"

        # 单水果模式下，顺序里只有 1 个 active fruit + 碗。
        # 例：['碗', '苹果'] 或 ['苹果', '碗']
        assert len(ack["pick_fruit_left_to_right_names"]) == 2
        assert set(ack["pick_fruit_left_to_right_leaves"]) == {
            "qsim_apple_2", "qsim_red_bowl",
        }
        print("single rollout fruits:", ack["pick_fruit_active_fruits"])
        print("single left-to-right:", ack["pick_fruit_left_to_right_names"])


asyncio.run(main())
```

单水果也可以用单元素 `pick_fruit_active_fruits` 表达，效果和 `pick_fruit_single_fruit` 一样：

```python
await ws.send(json.dumps({
    "type": "reset_env",
    "pick_fruit_active_fruits": ["qsim_apple_2"],
}))
```

当前单水果别名：`"apple"` -> `qsim_apple_2`，`"pumpkin"` -> `qsim_pumpkins`，`"carrot"` -> `qsim_carrot`。也可以直接传完整 leaf 名；100 个候选 fruit leaf 以 `pick_fruits_assets.yaml` 为准。

当前支持的 100 种水果 leaf：

```text
qsim_hy_xiangjiao_1
qsim_crown_pear_1
qsim_eagle_beak_peach_1
qsim_green_apple_1
qsim_yellow_lemon_1
qsim_star_fruit_1
qsim_murcott_1
qsim_apple_2
qsim_pumpkins
qsim_carrot
qsim_betel_nut_1
qsim_birchleaf_pear_1
qsim_black_cherry_1
qsim_blood_orange_1
qsim_brooks_cherry_1
qsim_green_tangerine_1
qsim_guiwei_lychee_1
qsim_honey_plum_1
qsim_kumquat_1
qsim_lime_1
qsim_lunwan_orange_1
qsim_prune_1
qsim_red_heart_papaya_1
qsim_rouge_apricot_1
qsim_snake_fruit_1
qsim_wax_apple_1
qsim_wild_peach_1
qsim_hy_suan_3
qsim_hy_tudou_3
qsim_hy_xianggu_1
qsim_hy_xianggu_2
qsim_apple_3
qsim_avocado_1
qsim_avocado_2
qsim_cucumber_1
qsim_cucumber_2
qsim_eggplant_1
qsim_eggplant_2
qsim_fig_1
qsim_flat_peach_1
qsim_garlic_1
qsim_garlic_2
qsim_ginger_1
qsim_ginger_2
qsim_green_daikon_1
qsim_guava_1
qsim_hy_daikon
qsim_kiwi_1
qsim_mango_1
qsim_mangosteen_1
qsim_onion_1
qsim_onion_2
qsim_onion_3
qsim_passion_1
qsim_pear_1
qsim_pear_2
qsim_pear_3
qsim_pear_4
qsim_persimmon_1
qsim_pomegranate_1
qsim_pumpkin_2
qsim_strawberry_1
qsim_sweet_pepper_1
qsim_sweet_potato_1
qsim_tomato_1
qsim_wax_apple_2
qsim_zucchini_1
qsim_apricot_1
qsim_asparagus_1
qsim_bamboo_shoot_1
qsim_bamboo_shoot_2
qsim_beet_1
qsim_bitter_melon_1
qsim_broccoli_1
qsim_cabbage_1
qsim_cabbage_2
qsim_cabbage_3
qsim_cauliflower_1
qsim_chayote_1
qsim_chili_1
qsim_corn_1
qsim_corn_2
qsim_corn_3
qsim_grape_1
qsim_grape_2
qsim_loofah_1
qsim_loofah_gourd_1
qsim_loquat_1
qsim_lotus_root_1
qsim_mango_2
qsim_mushroom_1
qsim_pitaya_1
qsim_plum_1
qsim_potato_2
qsim_sweet_pepper_2
qsim_sweet_pepper_3
qsim_taro_1
qsim_water_bamboo_1
qsim_water_chestnut_1
qsim_yam_1
```


#### 4.1.1 自定义水果初始场景

client 可以在新 rollout 开始时传 `pick_fruit_custom_initial_state`，一次性自定义 active fruits、碗位置/pose、每个水果位置/pose、机器人初始 root pose 和关节姿态。

坐标约定：水果和碗的 `xy` / `xyz` 是 `pick_fruit_layout` 里同一套桌面局部坐标；只写 `xy` 时 server 会保留资产原来的 `z`。机器人 `pos` 是 env-local root 位置，`quat_wxyz` 是 `(w,x,y,z)`；也可以用 `yaw_z_deg` 简写绕 Z 轴 yaw。`joint_positions` 推荐用 `{joint_name: value}`，joint 名以 `status.full_body_joint_names` 为准。

Custom 参数说明：

| 参数 | 是否必填 | 示例 | 说明 |
|---|---|---|---|
| `custom_policy` | 否，默认 `then_generalize` | `then_generalize` / `no_generalize` / `per_rollout` | 推荐使用的新字段；`per_rollout` 表示每个 rollout 都由 client 传一份新的 custom 初始状态 |
| `rollout_generalize` | 否，兼容旧写法 | `true` / `false` / `"per_rollout"` | 等价于 `custom_policy` 的简写：`true` -> `then_generalize`，`false` -> `no_generalize`，`"per_rollout"` -> `per_rollout` |
| `active_fruits` | 是，除非 `fruits` 里已经能推断出 active fruit | `["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"]` | 激活 1 个或 3 个水果；也可用别名 `apple` / `pumpkin` / `carrot` |
| `bowl` | 否 | `{"xy": [0.36, 0.14], "yaw_z_deg": 0.0}` | 自定义碗位置和 pose；位置可写 `xy` 或 `xyz`，姿态可写 `yaw_z_deg` 或 `quat_wxyz` |
| `fruits` | 否，但自定义水果位置/pose 时需要 | `{"qsim_apple_2": {"xy": [0.08, 0.05], "yaw_z_deg": 15.0}}` | 按 leaf 名给每个水果指定位置和 pose；未指定的 active fruit 会沿用 reset 后的默认/当前 pose |
| `robot.pos` | 否 | `[0.35, 1.80, 0.76]` | 自定义机器人初始 root 位置，env-local 坐标 |
| `robot.quat_wxyz` | 否 | `[0.7071, 0.0, 0.0, 0.7071]` | 自定义机器人初始 root 姿态；也可用 `robot.yaw_z_deg` |
| `robot.joint_positions` | 否 | `{"right_elbow_joint": -0.55}` | 自定义机器人初始关节姿态，只需要写要覆盖的关节 |

完整 request 示例：

```python
await ws.send(json.dumps({
    "type": "subscribe_step_result",
    "pick_fruit_custom_initial_state": {
        "custom_policy": "then_generalize",
        "rollout_generalize": True,
        "active_fruits": ["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"],
        "bowl": {"xy": [0.36, 0.14], "yaw_z_deg": 0.0},
        "fruits": {
            "qsim_apple_2": {"xy": [0.08, 0.05], "yaw_z_deg": 15.0},
            "qsim_pumpkins": {"xy": [0.24, 0.17], "yaw_z_deg": 80.0},
            "qsim_carrot": {"xy": [0.58, 0.07], "yaw_z_deg": 160.0},
        },
        "robot": {
            "pos": [0.35, 1.80, 0.76],
            "quat_wxyz": [0.7071, 0.0, 0.0, 0.7071],
            "joint_positions": {
                "right_shoulder_roll_joint": 0.12,
                "right_elbow_joint": -0.55,
                "left_shoulder_roll_joint": 0.12,
                "left_elbow_joint": -0.55,
            },
        },
    },
}))
```

选择 1：第一帧使用自定义初始状态，后续 rollout 继续按原水果泛化规则随机位置和 pose。若自定义里指定了 1 个或 3 个 active fruits，后续不带显式水果参数的 `reset_env` 会沿用这组 active fruits 做位置/pose 泛化。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8083"
CUSTOM_STATE = {
    "rollout_generalize": True,
    "active_fruits": ["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"],
    "bowl": {"xy": [0.36, 0.14], "yaw_z_deg": 0.0},
    "fruits": {
        "qsim_apple_2": {"xy": [0.08, 0.05], "yaw_z_deg": 15.0},
        "qsim_pumpkins": {"xy": [0.24, 0.17], "yaw_z_deg": 80.0},
        "qsim_carrot": {"xy": [0.58, 0.07], "yaw_z_deg": 160.0},
    },
    "robot": {
        "pos": [0.35, 1.80, 0.76],
        "quat_wxyz": [0.7071, 0.0, 0.0, 0.7071],
        "joint_positions": {
            "right_shoulder_roll_joint": 0.12,
            "right_elbow_joint": -0.55,
            "left_shoulder_roll_joint": 0.12,
            "left_elbow_joint": -0.55,
        },
    },
}


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "pick_fruit_custom_initial_state": CUSTOM_STATE,
        }))
        ack = json.loads(await ws.recv())
        init = json.loads(await ws.recv())
        assert ack["ok"], ack
        assert ack["pick_fruit_generalization_mode"] == "custom_initial_then_generalize"
        assert ack["pick_fruit_custom_rollout_generalize"] is True
        print("custom initial order:", ack["pick_fruit_left_to_right_names"])
        print("custom robot pose:", ack["pick_fruit_robot_pose"])

        # 下一次 rollout 不再传 custom；server 会沿用 active_fruits 做位置/pose 泛化。
        await ws.send(json.dumps({"type": "reset_env"}))
        ack2 = json.loads(await ws.recv())
        init2 = json.loads(await ws.recv())
        assert ack2["pick_fruit_generalization_mode"] in ("fixed_fruits", "single_fruit")
        print("generalized order:", ack2["pick_fruit_left_to_right_names"])


asyncio.run(main())
```

选择 2：第一帧使用自定义初始状态，后续 rollout 不泛化，重复恢复同一份自定义初始场景。client 第一次传 `custom_policy:"no_generalize"` 或 `rollout_generalize:false` 后，同一 worker 后续 `reset_env` 不带显式水果泛化参数即可复用该布局。

```python
CUSTOM_STATE_NO_GENERALIZE = {
    **CUSTOM_STATE,
    "custom_policy": "no_generalize",
    "rollout_generalize": False,
}

async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                              open_timeout=10, ping_interval=None) as ws:
    await ws.send(json.dumps({
        "type": "subscribe_step_result",
        "pick_fruit_custom_initial_state": CUSTOM_STATE_NO_GENERALIZE,
    }))
    ack = json.loads(await ws.recv())
    init = json.loads(await ws.recv())
    assert ack["pick_fruit_generalization_mode"] == "custom_initial_no_generalize"

    # 后续 rollout 不带 pick_fruit_active_fruits / pick_fruit_single_fruit / pick_fruit_generalize。
    # server 会返回 custom_initial_no_generalize_replay，并恢复同一份 custom 布局。
    await ws.send(json.dumps({"type": "reset_env"}))
    ack2 = json.loads(await ws.recv())
    init2 = json.loads(await ws.recv())
    assert ack2["pick_fruit_generalization_mode"] == "custom_initial_no_generalize_replay"
```

选择 3：每个 rollout 都由 client 自定义初始状态。client 每次 `subscribe_step_result` / `reset_env` 都传一份新的 `pick_fruit_custom_initial_state`，并设置 `custom_policy:"per_rollout"`。server 不会在 rollout 之间自动泛化，也不会复用上一份 custom；每次都以本次请求为准。

```python
CUSTOM_STATES = [
    {
        "custom_policy": "per_rollout",
        "active_fruits": ["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"],
        "bowl": {"xy": [0.36, 0.14], "yaw_z_deg": 0.0},
        "fruits": {
            "qsim_apple_2": {"xy": [0.08, 0.05], "yaw_z_deg": 15.0},
            "qsim_pumpkins": {"xy": [0.24, 0.17], "yaw_z_deg": 80.0},
            "qsim_carrot": {"xy": [0.58, 0.07], "yaw_z_deg": 160.0},
        },
        "robot": {"pos": [0.35, 1.80, 0.76], "quat_wxyz": [0.7071, 0.0, 0.0, 0.7071]},
    },
    {
        "custom_policy": "per_rollout",
        "active_fruits": ["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"],
        "bowl": {"xy": [0.22, 0.10], "yaw_z_deg": 7.0},
        "fruits": {
            "qsim_apple_2": {"xy": [0.50, 0.06], "yaw_z_deg": 21.0},
            "qsim_pumpkins": {"xy": [0.10, 0.17], "yaw_z_deg": 53.0},
            "qsim_carrot": {"xy": [0.62, 0.18], "yaw_z_deg": 107.0},
        },
        "robot": {"pos": [0.35, 1.80, 0.76], "quat_wxyz": [0.7071, 0.0, 0.0, 0.7071]},
    },
]

async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                              open_timeout=10, ping_interval=None) as ws:
    # 第 0 个 rollout：订阅并发送第一份 custom。
    await ws.send(json.dumps({
        "type": "subscribe_step_result",
        "pick_fruit_custom_initial_state": CUSTOM_STATES[0],
    }))
    ack = json.loads(await ws.recv())
    init = json.loads(await ws.recv())
    assert ack["pick_fruit_generalization_mode"] == "custom_initial_per_rollout"
    assert ack["pick_fruit_custom_policy"] == "per_rollout"

    # 后续每个 rollout：reset_env 时传新的 custom。
    for custom_state in CUSTOM_STATES[1:]:
        await ws.send(json.dumps({
            "type": "reset_env",
            "pick_fruit_custom_initial_state": custom_state,
        }))
        ack = json.loads(await ws.recv())
        init = json.loads(await ws.recv())
        assert ack["pick_fruit_generalization_mode"] == "custom_initial_per_rollout"
        assert ack["pick_fruit_custom_policy"] == "per_rollout"
        print("per-rollout custom order:", ack["pick_fruit_left_to_right_names"])
```

#### 4.1.2 client 读取“从左到右”顺序

每次新 rollout 开始后，client 直接从 ack 里读取顺序即可：

```python
ack = json.loads(await ws.recv())   # subscribe_step_result_response 或 reset_env_response
init = json.loads(await ws.recv())  # phase=initial step_result

left_to_right_names = ack["pick_fruit_left_to_right_names"]
left_to_right_leaves = ack["pick_fruit_left_to_right_leaves"]
left_to_right_order = ack["pick_fruit_left_to_right_order"]

# 例：['苹果', '南瓜', '碗', '胡萝卜']
print("从左到右中文名:", left_to_right_names)

# 例：['qsim_apple_2', 'qsim_pumpkins', 'qsim_red_bowl', 'qsim_carrot']
print("从左到右 leaf:", left_to_right_leaves)

# 完整对象里有 kind / leaf / display_name / x / y / xy，方便调试或落盘。
for obj in left_to_right_order:
    print(obj["kind"], obj["display_name"], obj["leaf"], obj["xy"])
```

如果同一条连接里跑下一次 rollout，用 `reset_env` 后按同样方式读：

```python
await ws.send(json.dumps({
    "type": "reset_env",
    # 固定三水果：写 3 个 pick_fruit_active_fruits；
    # 随机三水果：删掉 pick_fruit_active_fruits；
    # 单水果：写 pick_fruit_single_fruit，或只写 1 个 pick_fruit_active_fruits；
    # 每轮 custom：改为写 pick_fruit_custom_initial_state。
    "pick_fruit_active_fruits": ["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"],
}))

ack = json.loads(await ws.recv())   # reset_env_response
init = json.loads(await ws.recv())  # 已订阅时 server 会继续推 phase=initial

order_for_policy = ack["pick_fruit_left_to_right_names"]
```

顺序规则：返回列表包含当前 active fruits 和 `qsim_red_bowl`，按画面左到右排序；三水果模式是 3 个水果 + 碗，单水果模式是 1 个水果 + 碗；当前协议里 `pick_fruit_left_to_right_axis == "tabletop_local_x_asc"`。

要点：

- `await ws.send(...)` + `await ws.recv()` 严格按 `submit -> ack -> push` 时序；不要用 `asyncio.gather` 并发收发，否则会乱
- 同一条 WS 连接里跑下一次 rollout 时，先再发一次 `subscribe_step_result`；`pick_fruit` worker 默认把每次订阅当作 rollout reset + 泛化边界（返回 `rollout_reset_applied=true`）
- `pick_fruit` 泛化模式：带 3 个 `"pick_fruit_active_fruits"` 时固定这 3 种水果、只随机位置和 pose；不带该字段时从当前可解析水果池里选 3 个，组合、位置、pose 都随机；带 `"pick_fruit_single_fruit": "apple"` 或单元素 `"pick_fruit_active_fruits": ["qsim_apple_2"]` 时只激活该水果 + 碗，并随机位置和 pose；带 `"pick_fruit_custom_initial_state"` 时可自定义碗/水果/机器人初始状态，并通过 `custom_policy` 选择后续 rollout 行为；带 `"pick_fruit_negative_sample_type"` 时生成负样例；需要关闭普通泛化时传 `"pick_fruit_generalize": false`；需要 rollout 之间随机机器人手臂/手指初始关节时额外传 `"pick_fruit_robot_generalize": true`
- 每次 pick_fruit rollout reset 后，`subscribe_step_result_response` / `reset_env_response` / `status_response` / 初始 `step_result` 都会带 `pick_fruit_generalization_mode`、`pick_fruit_negative_sample_type`、`pick_fruit_robot_generalize`、`pick_fruit_robot_pose`、`pick_fruit_left_to_right_names` 和 `pick_fruit_left_to_right_leaves`；非空桌面顺序包含当前 active fruits + `qsim_red_bowl`，按桌面局部 `X` 从小到大排序（`pick_fruit_left_to_right_axis="tabletop_local_x_asc"`），即相机/机器人视角从左到右。
- 如果你想保持订阅不变、只显式开新 rollout，可发 `{"type":"reset_env"}`；若当前已订阅，server 会回 `reset_env_response`，随后主动推新的 `phase=initial`
- `submit_actions.actions` 内层长度必须 == `status.action_dim`，否则 server 会回 error response 并丢弃
- 如果模型 cfg 是 36 维但要喂 arm26 worker，可以用 `g1o6_action_layout.compact_full36_to_arm26` 在 client 端 gather
- `step_result.frames` 一帧约 30–80 KB（jpeg q=85，720p）；30 帧批次约 1–2 MB，远低于 ws max_size 32 MB

### 4.2 pick_fruit 负样例用法示例

负样例也是 rollout 级机制，不需要 `switch_pick_fruit_layout`，也不会写 `/tmp/g1o6_pick_fruit_layout_<port>`。推荐在新 rollout 开始时通过 `reset_env` 或 `subscribe_step_result` 传 `pick_fruit_negative_sample_type`：

| 类型 | 画面语义 | 典型返回 |
|---|---|---|
| `fruits_in_bowl` | 水果已经放入碗中 | `pick_fruit_generalization_mode="negative_fruits_in_bowl"`，active fruits 仍返回本次水果 |
| `no_fruit` | 没有水果，碗还在桌面上 | `pick_fruit_active_fruits=[]`，left-to-right 通常只包含碗 |
| `empty_table` | 台面空无一物 | `pick_fruit_active_fruits=[]`，碗隐藏，`pick_fruit_left_to_right_*` 为空 |

最小 Python 示例：

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8083"


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "reset_env",
            "pick_fruit_negative_sample_type": "fruits_in_bowl",
        }))

        ack = json.loads(await ws.recv())
        assert ack["ok"], ack
        assert ack["pick_fruit_generalization_mode"] == "negative_fruits_in_bowl"
        assert ack["pick_fruit_negative_sample_type"] == "fruits_in_bowl"
        print(ack["pick_fruit_active_fruits"])
        print(ack["pick_fruit_left_to_right_names"])


asyncio.run(main())
```

三种负样例可以直接换字段值：

```jsonc
{"type": "reset_env", "pick_fruit_negative_sample_type": "fruits_in_bowl"}
{"type": "reset_env", "pick_fruit_negative_sample_type": "no_fruit"}
{"type": "reset_env", "pick_fruit_negative_sample_type": "empty_table"}
```

也支持放在泛化对象里，适合把三果/单果和负样例参数统一到一个配置结构：

```jsonc
{
  "type": "subscribe_step_result",
  "pick_fruit_generalization": {
    "negative_sample_type": "empty_table"
  }
}
```

兼容别名：`fruit_in_bowl` / `placed_in_bowl` / `in_bowl` -> `fruits_in_bowl`，`no_fruits` / `hide_fruit` / `empty_fruits` -> `no_fruit`，`table_empty` / `hide_fruit_bowl` -> `empty_table`。

### 4.3 water_plant 两种泛化模式示例

可选对象 leaf：

- 水壶：`qsim_hy_watering_can`, `qsim_water_pot_3`, `qsim_water_pot_2`
- 花盆：`model_pottedplant`, `qsim_potted_plant_1`, `qsim_potted_plant_2`

模式 1：client 指定水壶和花盆；后续 rollout 固定这两个对象，只随机 slot、XY 抖动、pose/yaw 和机器人初始 pose。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8090"  # 调试 worker；正式使用时换成目标 worker
FIXED_OBJECTS = {
    "watering_can": "qsim_water_pot_3",
    "potted_plant": "qsim_potted_plant_2",
}


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "water_plant_active_objects": FIXED_OBJECTS,
        }))

        ack = json.loads(await ws.recv())   # subscribe_step_result_response
        init = json.loads(await ws.recv())  # phase=initial step_result

        assert ack["ok"], ack
        assert ack["water_plant_generalization_mode"] == "fixed_objects"
        assert ack["water_plant_active_objects"] == FIXED_OBJECTS
        assert init["phase"] == "initial"
        print("fixed water_plant objects:", ack["water_plant_active_objects"])


asyncio.run(main())
```

已经订阅后开下一次 rollout，字段完全一样：

```python
await ws.send(json.dumps({
    "type": "reset_env",
    "water_plant_active_objects": FIXED_OBJECTS,
}))
ack = json.loads(await ws.recv())   # reset_env_response
init = json.loads(await ws.recv())  # 如果当前连接已订阅，会继续推 phase=initial
```

模式 2：client 不指定水壶/花盆；server 每次 rollout 自动水壶 3 选 1、花盆 3 选 1，并随机位置和 pose。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8090"


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "subscribe_step_result"}))

        ack = json.loads(await ws.recv())   # subscribe_step_result_response
        init = json.loads(await ws.recv())  # phase=initial step_result

        assert ack["ok"], ack
        assert ack["water_plant_generalization_mode"] == "random_objects"
        assert ack["water_plant_active_objects"]["watering_can"]
        assert ack["water_plant_active_objects"]["potted_plant"]
        assert init["phase"] == "initial"
        print("random water_plant objects:", ack["water_plant_active_objects"])


asyncio.run(main())
```

要点：

- `water_plant_active_objects` 也可以写成两元素 list，例如 `["qsim_water_pot_3", "qsim_potted_plant_2"]`，顺序不限，但必须一水壶一花盆。
- 需要关闭 rollout 泛化时传 `"water_plant_generalize": false`。
- 位置/pose 参数与 `easim_ck` 浇水场景 N-key 路径一致：两个桌面 slot、同样 XY jitter；水壶 yaw `[-45, 45]`，花盆 yaw `[0, 360]`；机器人关节、位置、yaw 也使用同一组随机范围。


---

### 4.4 stack_plate_bowl 两种 rollout 泛化模式示例

`stack_plate_bowl` 的运行期泛化不走 hotkey，也不会写 `pick_fruit` 的启动期 layout state file。每个 `subscribe_step_result` 或 `reset_env` 都是新的 rollout 边界：server 会按 `easim_ck` 同一套参数重新采样 Area_6 桌面局部 XY 和 yaw，避免 plate/bowl 重叠。

资产池来自 `source/easim/scenario_loader/configs/stack_plate_bowl_assets.yaml`：当前 6 个 plate、8 个 bowl。YAML `default_visible` 是 `qsim_hy_19plate`、`qsim_hy_panzi_3`、`qsim_bowl_1`。

模式 1：默认初始化。client 不指定物体时，使用 YAML 里的 2 个默认 plate + 1 个默认 bowl；每个 rollout 仍会随机位置和 yaw。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8080"


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "stack_plate_bowl_generalization": {"mode": "default"},
        }))

        ack = json.loads(await ws.recv())   # subscribe_step_result_response
        init = json.loads(await ws.recv())  # phase=initial step_result

        assert ack["ok"], ack
        assert ack["stack_plate_bowl_generalization_mode"] == "default_visible"
        print(ack["stack_plate_bowl_active_objects"])
        print(ack["stack_plate_bowl_layout"])
        assert init["phase"] == "initial"


asyncio.run(main())
```

模式 2：随机泛化。server 每个 rollout 随机选择 2 个不同 plate + 1 个 bowl，再随机桌面位置和 yaw。

```jsonc
{
  "type": "reset_env",
  "stack_plate_bowl_generalization": {"mode": "random"}
}
```

模式 3：指定 plate/bowl。组合固定，位置和 yaw 每个 rollout 泛化；也可以传 `plate_xyz` / `bowl_xyz` 固定某些物体的位置。

```jsonc
{
  "type": "reset_env",
  "stack_plate_bowl_generalization": {
    "mode": "fixed",
    "active_plates": ["qsim_hy_19plate", "qsim_hy_panzi_3"],
    "active_bowls": ["qsim_bowl_1"]
  }
}
```

指定位置示例：`xyz` 是桌面局部坐标；没有给 yaw/quat 时，yaw 仍会按泛化逻辑随机。需要固定 yaw 时传 `plate_yaw_z_deg` / `bowl_yaw_z_deg`，或直接传 `plate_orient_quat_wxyz` / `bowl_orient_quat_wxyz`。

```jsonc
{
  "type": "reset_env",
  "stack_plate_bowl_generalization": {
    "mode": "fixed",
    "active_plates": ["qsim_hy_19plate", "qsim_hy_panzi_3"],
    "active_bowls": ["qsim_bowl_1"],
    "plate_xyz": {
      "qsim_hy_19plate": [0.14, 0.045, 0.0],
      "qsim_hy_panzi_3": [0.42, 0.175, 0.0]
    },
    "bowl_xyz": {
      "qsim_bowl_1": [0.62, 0.095, 0.0]
    },
    "plate_yaw_z_deg": {
      "qsim_hy_19plate": 15.0,
      "qsim_hy_panzi_3": 120.0
    },
    "bowl_yaw_z_deg": {
      "qsim_bowl_1": 270.0
    }
  }
}
```


ack 中常用字段：

```jsonc
"stack_plate_bowl_generalization_mode": "default_visible" | "random_objects" | "fixed_objects" | "default_visible_custom_pose",
"stack_plate_bowl_active_objects": {"plates": ["...", "..."], "bowls": ["..."]},
"stack_plate_bowl_layout": [...]
```


### 4.5 pick_paper_balls 四种 rollout 泛化模式示例

`pick_paper_balls` 的运行期泛化不走 hotkey，也不会写 `pick_fruit` 的启动期 layout state file。每个 `subscribe_step_result` 或 `reset_env` 都是新的 rollout 边界：server 会按 `easim_ck` 同一套参数在 Area_7 桌面上重新采样物体 XY 和 yaw。

资产池来自 `source/easim/scenario_loader/configs/pick_paper_balls_assets.yaml`：当前 3 个 orange peel、3 个 paper ball、3 个 desk trash can。YAML `default_visible` 是 `qsim_orange_peel`、`qsim_paper_ball`、`qsim_desk_trash_can_1`。

当前 client 可选的资产 leaf 名称如下；`fixed` 模式的 `active_objects` 和 `custom` 模式每个 item 的 `leaf` 都使用这些字符串：

```jsonc
{
  "orange_peels": [
    "qsim_orange_peel",
    "qsim_orange_peel_2",
    "qsim_orange_peel_3"
  ],
  "paper_balls": [
    "qsim_paper_ball",
    "qsim_paper_ball_2",
    "qsim_paper_ball_3"
  ],
  "trash_cans": [
    "qsim_desk_trash_can_1",
    "qsim_desk_trash_can_2",
    "qsim_desk_trash_can_3"
  ]
}
```

模式 1：默认 visible。client 指定 `mode: "default"` 时，使用 YAML 里的默认橘皮 + 默认纸团 + 默认垃圾桶；每个 rollout 仍会随机桌面位置和 yaw。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8080"


async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "pick_paper_balls_generalization": {"mode": "default"},
        }))

        ack = json.loads(await ws.recv())   # subscribe_step_result_response
        init = json.loads(await ws.recv())  # phase=initial step_result

        assert ack["ok"], ack
        assert ack["pick_paper_balls_generalization_mode"] == "default_visible"
        print(ack["pick_paper_balls_active_objects"])
        print(ack["pick_paper_balls_layout"])
        assert init["phase"] == "initial"


asyncio.run(main())
```

模式 2：随机泛化。server 每个 rollout 随机选择 1 个橘皮 + 1 个纸团 + 1 个垃圾桶，再随机桌面位置和 yaw。当前 `pick_paper_balls` 如果不传泛化字段，也会按随机模式处理；显式写 `mode: "random"` 更清楚。

```jsonc
{
  "type": "reset_env",
  "pick_paper_balls_generalization": {"mode": "random"}
}
```

也可以用顶层字段表达随机：

```jsonc
{
  "type": "reset_env",
  "pick_paper_balls_active_objects": "random"
}
```

模式 3：指定物体种类。组合固定，位置和 yaw 每个 rollout 继续泛化。固定模式必须同时指定 1 个橘皮、1 个纸团和 1 个垃圾桶；当前协议不支持只固定其中一类、其余类别由 server 随机。

```jsonc
{
  "type": "reset_env",
  "pick_paper_balls_generalization": {
    "mode": "fixed",
    "active_objects": {
      "orange_peel": "qsim_orange_peel_2",
      "paper_ball": "qsim_paper_ball_3",
      "trash_can": "qsim_desk_trash_can_2"
    }
  }
}
```

等价的顶层写法：

```jsonc
{
  "type": "reset_env",
  "pick_paper_balls_orange_peel": "qsim_orange_peel_2",
  "pick_paper_balls_paper_ball": "qsim_paper_ball_3",
  "pick_paper_balls_trash_can": "qsim_desk_trash_can_2"
}
```

模式 4：client 自定义多物体 layout。数量上限等于 YAML 里已有资产数量：当前最多 3 个橘皮、3 个纸团、3 个垃圾桶；同一类别内不允许重复使用同一个 leaf。每个 item 可以只写 `leaf`，也可以写 `xy` / `xyz` 和 `yaw_z_deg` / `quat_wxyz`；没写位置时会恢复该 leaf 在 USD 里的原始桌面位置。

```jsonc
{
  "type": "reset_env",
  "pick_paper_balls_generalization": {
    "mode": "custom",
    "orange_peels": [
      {"leaf": "qsim_orange_peel", "xy": [0.12, 0.05], "yaw_z_deg": 30.0},
      {"leaf": "qsim_orange_peel_2", "xyz": [0.30, 0.16, 0.0]}
    ],
    "paper_balls": [
      {"leaf": "qsim_paper_ball_2", "xy": [0.42, 0.08]},
      {"leaf": "qsim_paper_ball_3", "xy": [0.55, 0.18], "yaw_z_deg": 120.0}
    ],
    "trash_cans": [
      {"leaf": "qsim_desk_trash_can_1", "xy": [0.62, 0.12]},
      {"leaf": "qsim_desk_trash_can_3", "xy": [0.18, 0.18]}
    ]
  }
}
```

等价地，也可以把 custom layout 放在顶层 `pick_paper_balls_custom_initial_state`：

```jsonc
{
  "type": "reset_env",
  "pick_paper_balls_custom_initial_state": {
    "orange_peels": {
      "qsim_orange_peel": [0.12, 0.05],
      "qsim_orange_peel_3": {"xy": [0.30, 0.16], "yaw_z_deg": 45.0}
    },
    "paper_balls": ["qsim_paper_ball"],
    "trash_cans": [
      {"leaf": "qsim_desk_trash_can_2", "xy": [0.62, 0.12]}
    ]
  }
}
```

custom 模式允许某一类为空，例如 `"orange_peels": []` 表示本 rollout 不显示果皮；未出现的类别也按空列表处理。需要继续随机种类/位置时，不要用 custom，改用 `mode: "random"`。

ack 中常用字段：

```jsonc
"pick_paper_balls_generalization_mode": "default_visible" | "random_objects" | "fixed_objects" | "custom_objects",
"pick_paper_balls_active_objects": {
  "orange_peel": "...",   // 兼容旧 client：第一项，没有则不存在
  "paper_ball": "...",
  "trash_can": "...",
  "orange_peels": ["..."],
  "paper_balls": ["..."],
  "trash_cans": ["..."]
},
"pick_paper_balls_layout": [...],
"pick_paper_balls_physx_synced_bodies": 0
```

需要关闭 rollout 泛化时传 `"pick_paper_balls_generalize": false`。这种情况下 server 只做普通 reset，不重新选择物体/位置；一般只在调试时使用。

### 4.6 pick_fruits_and_paper_balls 组合整理场景示例

`pick_fruits_and_paper_balls` 是一个组合桌面场景：每个 rollout 同时放置 3 个水果、1 个碗/盘、1 个果皮、1 个纸团和 1 个垃圾桶。它复用 `pick_fruit` 的水果桌面层和 `pick_paper_balls` 的纸团/垃圾桶资产，但在运行期按同一张桌面做组合泛化。

当前 rollout 协议刻意保持简洁，只支持两种模式：

| 模式 | 请求写法 | 行为 |
|---|---|---|
| 默认初始化 | `"mode": "default"` 或 `"mode": "default_visible"` | 使用 YAML/default layout 的默认组合；每个 rollout 仍会重置到组合场景的默认可见布局 |
| 随机泛化 | `"mode": "random"` 或 `"mode": "random_objects"` | 在 rollout 边界同时重采样水果、碗/盘、果皮、纸团、垃圾桶的种类、桌面位置和 yaw |

如果不传 `pick_fruits_and_paper_balls_generalization`，当前 server 会按随机模式处理。除了主字段外，也兼容两个顶层别名：

```jsonc
{
  "pick_fruits_and_paper_balls_mode": "random",
  "pick_fruits_and_paper_balls_generalize": true
}
```

当前 **不支持** `fixed` / `custom`。如果 client 传了这两类 mode，server 会直接返回错误。

默认布局示例：

```jsonc
{
  "type": "reset_env",
  "pick_fruits_and_paper_balls_generalization": {"mode": "default"}
}
```

随机泛化示例：

```jsonc
{
  "type": "subscribe_step_result",
  "force_reset": true,
  "pick_fruits_and_paper_balls_generalization": {"mode": "random"}
}
```

组合场景有一个和 `pick_paper_balls` / `pour_coffee` 不同的小点：`subscribe_step_result_response` 和 `reset_env_response` 当前**不会**回显 `pick_fruits_and_paper_balls_active_objects` / `pick_fruits_and_paper_balls_layout`。client 需要从后续 `step_result` 或显式 `status` 里读取。

`step_result` / `status_response` 中常用字段：

```jsonc
"pick_fruits_and_paper_balls_generalization_mode": "default_visible" | "random_objects",
"pick_fruits_and_paper_balls_active_objects": {
  "fruits": ["...", "...", "..."],
  "bowl": "qsim_red_bowl",
  "orange_peel": "...",
  "paper_ball": "...",
  "trash_can": "..."
},
"pick_fruits_and_paper_balls_layout": [
  {
    "env_id": 0,
    "pick_fruits": {...},
    "paper_ball_cleanup": {...}
  }
]
```

其中 `pick_fruits_and_paper_balls_layout` 是组合快照：`pick_fruits` 部分对应水果/碗布局，`paper_ball_cleanup` 部分对应果皮/纸团/垃圾桶布局。

完整 client 示例：切到 `pick_fruits_and_paper_balls`，开始一个随机 rollout，保存首帧，并执行 5 次闭环 action。真实策略只需要替换 `policy()`。

```python
import asyncio
import base64
import json
from pathlib import Path

import websockets

URI = "ws://172.31.0.226:8083"
SCENE_ID = "pick_fruits_and_paper_balls"
OUT_DIR = Path("pick_fruits_and_paper_balls_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_b64_image(image_obj, path: Path):
    path.write_bytes(base64.b64decode(image_obj["data"]))


async def recv_type(ws, expected_types, timeout=240):
    expected = set(expected_types)
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") in expected:
            return msg


async def get_status(uri: str) -> dict:
    async with websockets.connect(uri, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "status"}))
        return await recv_type(ws, {"status_response"}, timeout=30)


async def ensure_scene(uri: str, scene_id: str) -> dict:
    status = await get_status(uri)
    if status.get("scene_id") == scene_id:
        return status

    async with websockets.connect(uri, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "switch_scene", "scene_id": scene_id}))
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except Exception:
            pass

    for _ in range(80):
        await asyncio.sleep(3)
        try:
            status = await get_status(uri)
        except Exception:
            continue
        if status.get("scene_id") == scene_id:
            return status
    raise TimeoutError(f"worker did not switch to {scene_id}")


def policy(step_result: dict, action_dim: int) -> list[list[float]]:
    return [[0.0] * action_dim for _ in range(30)]


async def run_rollout():
    status = await ensure_scene(URI, SCENE_ID)
    action_dim = int(status["action_dim"])
    print("scene", status["scene_id"], "layout", status["action_layout"], "dim", action_dim)

    async with websockets.connect(URI, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
            "pick_fruits_and_paper_balls_generalization": {
                "mode": "random",
            },
        }))

        ack = await recv_type(ws, {"subscribe_step_result_response"})
        assert ack.get("ok"), ack
        print("subscribe ack ok")

        step_result = await recv_type(ws, {"step_result"})
        assert step_result.get("phase") == "initial", step_result.get("phase")

        print("mode", step_result.get("pick_fruits_and_paper_balls_generalization_mode"))
        print("active", step_result.get("pick_fruits_and_paper_balls_active_objects"))
        print("layout", step_result.get("pick_fruits_and_paper_balls_layout"))

        frame0 = step_result["frames"][0]
        save_b64_image(frame0["image"], OUT_DIR / "initial.jpg")

        for step_idx in range(5):
            await ws.send(json.dumps({
                "type": "submit_actions",
                "actions": policy(step_result, action_dim),
            }))
            submit_ack = await recv_type(ws, {"submit_actions_response"})
            assert submit_ack.get("ok"), submit_ack
            step_result = await recv_type(ws, {"step_result"})
            last_frame = step_result["frames"][-1]
            save_b64_image(last_frame["image"], OUT_DIR / f"step{step_idx:02d}.jpg")
            print(
                "step",
                step_idx,
                "executed",
                step_result.get("executed_actions"),
                "active",
                step_result.get("pick_fruits_and_paper_balls_active_objects"),
            )


asyncio.run(run_rollout())
```

需要关闭 rollout 泛化时传 `"pick_fruits_and_paper_balls_generalize": false`。这种情况下 server 只做普通 reset，不重新重采样组合布局。

### 4.6.1 pick_fruits_and_paper_balls_showroom Showroom 复杂桌面整理场景

`pick_fruits_and_paper_balls_showroom` 使用独立资产：

`/root/code/easim/assets/environment/Showroom/scene_1/scene_1.usd`

场景中的固定对象是：

- `qsim_carrot`
- `qsim_pumpkins`
- `apple_3`
- `qsim_red_bowl`
- `qsim_desk_trash_can_1`
- `qsim_paper_ball`
- `qsim_orange_peel`

这个 scene ID 与 `pick_fruits_and_paper_balls` 隔离。Showroom 的桌子在世界原点附近，桌面高度约为 `0.756 m`；机器人默认位于桌子 `-Y` 一侧，root pose 为：

```json
{
  "pos": [0.0, -0.65, 0.76],
  "rot_wxyz": [0.7071, 0.0, 0.0, 0.7071]
}
```

当前支持固定布局模式。每次 `subscribe_step_result` / `reset_env` 会恢复 USD 原始布局；不要传 `pick_fruits_and_paper_balls_generalization`，也不要依赖 Room01 的 Area_2 / Area_7 路径。当前不支持 Showroom 的对象种类、位置、yaw 自定义和自动 `objects_in_container` 评测，避免把 Room01 坐标泛化逻辑误用到 `scene_1`。

完整 client smoke 示例：

```python
import asyncio
import base64
import json
from pathlib import Path

import websockets

URI = "ws://172.31.0.226:8080"
OUT_DIR = Path("showroom_scene1_rollout")
OUT_DIR.mkdir(exist_ok=True)


async def recv_type(ws, expected, timeout=120):
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") in expected:
            return msg


async def main():
    async with websockets.connect(
        URI, max_size=64 * 1024 * 1024, ping_interval=None
    ) as ws:
        await ws.send(json.dumps({"type": "status"}))
        status = await recv_type(ws, {"status_response"})
        assert status["scene_id"] == "pick_fruits_and_paper_balls_showroom"

        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True
        }))
        ack = await recv_type(ws, {"subscribe_step_result_response"})
        assert ack["ok"], ack

        result = await recv_type(ws, {"step_result"})
        frame0 = result["frames"][0]
        (OUT_DIR / "initial.jpg").write_bytes(
            base64.b64decode(frame0["image"]["data"])
        )

        action_dim = int(result["action_dim"])
        actions = [[0.0] * action_dim for _ in range(30)]
        await ws.send(json.dumps({
            "type": "submit_actions",
            "actions": actions
        }))
        await recv_type(ws, {"submit_actions_response"})
        result = await recv_type(ws, {"step_result"})
        (OUT_DIR / "final.jpg").write_bytes(
            base64.b64decode(result["frames"][-1]["image"]["data"])
        )


asyncio.run(main())
```

### 4.6.2 Showroom O6 scene_2/3/4/7/8/9/10/11 完整切换与 rollout 示例

这些场景使用以下独立 `scene_id`：

```text
showroom_scene_2   showroom_scene_3   showroom_scene_4
showroom_scene_7   showroom_scene_8   showroom_scene_9
showroom_scene_10  showroom_scene_11
```

它们统一使用 G1+O6 `arm26/full36` action 和单目 `viewport_cam`。除 `showroom_scene_11` 外，其余场景使用固定 USD 布局；不要给这些 scene 传 `pick_fruit_*`、`pick_paper_balls_*` 或 `pour_coffee_*` 泛化字段。Scene11 使用专属的 `showroom_scene_11_generalization`，当前仍没有目标单词或字母顺序的自动 success/pass_rate 评测。

下面的完整示例会切到目标场景，等待 worker 自重启，开始一个 rollout，保存首帧并发送 2 个 action。把 `SCENE_ID` 改成上面任意一个 ID 即可：

```python
import asyncio
import base64
import json
import time
from pathlib import Path

import websockets

URI = "ws://172.31.0.226:8081"
SCENE_ID = "showroom_scene_7"
OUT = Path(f"{SCENE_ID}_rollout")
OUT.mkdir(exist_ok=True)


async def recv_type(ws, expected, timeout=180):
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") in expected:
            return msg


async def status():
    async with websockets.connect(URI, open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "status"}))
        return await recv_type(ws, {"status_response"}, 20)


async def ensure_scene():
    current = await status()
    if current["scene_id"] != SCENE_ID:
        async with websockets.connect(URI, open_timeout=10, ping_interval=None) as ws:
            await ws.send(json.dumps({"type": "switch_scene", "scene_id": SCENE_ID}))
            ack = await recv_type(ws, {"switch_scene_response"}, 20)
            assert ack["ok"], ack

        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            try:
                current = await status()
                if current["scene_id"] == SCENE_ID:
                    return current
            except Exception:
                pass
            await asyncio.sleep(5)
        raise TimeoutError(f"worker did not switch to {SCENE_ID}")
    return current


async def main():
    st = await ensure_scene()
    action_dim = int(st["action_dim"])
    async with websockets.connect(
        URI, max_size=64 * 1024 * 1024, ping_interval=None
    ) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True
        }))
        ack = await recv_type(ws, {"subscribe_step_result_response"})
        assert ack["ok"], ack
        initial = await recv_type(ws, {"step_result"})
        frame0 = initial["frames"][0]
        (OUT / "initial.jpg").write_bytes(
            base64.b64decode(frame0["image"]["data"])
        )

        qpos51 = frame0["state"]["joint_position"]
        full_index = {
            name: index for index, name in enumerate(st["full_body_joint_names"])
        }
        action_names = (
            st.get("action_resolved_joint_names") or st["action_joint_names"]
        )
        hold_action = [
            float(qpos51[full_index[name]]) for name in action_names
        ]
        assert len(hold_action) == action_dim

        await ws.send(json.dumps({
            "type": "submit_actions",
            "actions": [hold_action for _ in range(2)]
        }))
        submit_ack = await recv_type(ws, {"submit_actions_response"})
        assert submit_ack["ok"], submit_ack
        result = await recv_type(ws, {"step_result"})
        (OUT / "final.jpg").write_bytes(
            base64.b64decode(result["frames"][-1]["image"]["data"])
        )


asyncio.run(main())
```

#### Scene11 字母 rollout 泛化

`showroom_scene_11` 和 `showroom_scene_11_wuji` 共用以下四种模式：

| mode | 字母 | 位置 | 颜色/朝向/桌面 |
|---|---|---|---|
| `single_custom` | `letter` 可选，默认 `A` | `position: [x, y]` 可选，默认该字母 USD authored 位置 | `cube_color` / `letter_color` 可选，默认原灰色方块/蓝色字母 |
| `single_random` | `randomize_letter` 独立开关 | `randomize_position` 独立开关 | `randomize_color` 独立开关 |
| `thirteen_custom` | `letters` 可选；必须恰好 13 个不重复 A-Z，默认 A-M | `randomize_position`，默认 `true` | `randomize_color`、`randomize_yaw`、`randomize_table_color`，默认都为 `true` |
| `thirteen_random` | server 每个 rollout 从 tmp 布局选择 13 个不重复 A-Z | `randomize_position`，默认 `true` | `randomize_color`、`randomize_yaw`、`randomize_table_color`，默认都为 `true` |

13 字母模式与 `/root/code/tmp/easim` 的预生成数据完全对齐，不重新在线采样。每个 rollout 从 `source/easim/scenes/showroom/scene11/layouts/anchor_*.jsonl` 中选择一条布局：共 26 个 anchor，每个 anchor 20 条布局，每条布局 13 个不重复字母，三排数量固定为从上到下 `4 / 4 / 5`。桌面局部范围为 `x ∈ [-0.24, 0.24] m`、`y ∈ [-0.09, 0.09] m`，等效半径 `0.03 m`，额外间距 `0.01 m`，yaw 范围 `[-15°, 15°]`。

13 字母模式的泛化开关均为 rollout 级别：`randomize_position` 控制 tmp 位置，`randomize_yaw` 控制 tmp yaw，`randomize_color` 控制积木本体颜色，`randomize_table_color` 控制桌面颜色。四项默认都开启，也可以分别设置为 `false`。**字体颜色固定黑色 `[0.0, 0.0, 0.0]`，字体大小固定 `scale=1.0`，不支持字体颜色或大小泛化。**

颜色字段使用归一化 RGB，不使用颜色名称：`cube_color: [R, G, B]` 设置方块本体颜色，`letter_color: [R, G, B]` 设置方块正面的字母颜色；每个通道都必须是 `[0.0, 1.0]` 内的数值，因此支持任意 RGB 颜色。常用写法如下：

| 颜色 | RGB |
|---|---|
| 黑色 | `[0.0, 0.0, 0.0]` |
| 白色 | `[1.0, 1.0, 1.0]` |
| 红色 | `[1.0, 0.0, 0.0]` |
| 绿色 | `[0.0, 1.0, 0.0]` |
| 蓝色 | `[0.0, 0.0, 1.0]` |
| 黄色 | `[1.0, 1.0, 0.0]` |

颜色行为和限制：

- `single_custom` 和 `single_random` 都可以传 `cube_color` / `letter_color`。两个字段均可单独省略；省略时分别使用默认灰色方块 `[0.780392, 0.780392, 0.780392]` 和默认蓝色字母 `[0.0, 0.32, 1.0]`。
- `single_random` 设置 `randomize_color: true` 时，server 会在每个 rollout 中分别随机方块颜色和字母颜色，随机结果优先于显式 `cube_color` / `letter_color`；每个通道的随机范围是 `[0.15, 0.90]`，不会产生纯黑或纯白。
- `thirteen_custom` 和 `thirteen_random` 不接受逐字母 `cube_color` / `letter_color`。`randomize_color: true` 时使用 tmp 布局中的固定 8 色 palette 的积木颜色；`false` 时恢复 authored 的灰色积木。字体在两种情况下都固定为黑色。
- `randomize_table_color: true` 时使用 tmp 布局中的固定 8 色桌面 palette；设为 `false` 时恢复 USD authored 桌面材质。`randomize_yaw: false` 时恢复 authored 朝向；`randomize_position: false` 时恢复 authored 位置。

省略整个 `showroom_scene_11_generalization` 字段时保持向后兼容，使用 USD authored 的全部 26 个字母。下面是一个可直接用于 O6 或 Wuji Scene11 的完整 rollout 请求；Wuji 只需把 worker 预先切到 `showroom_scene_11_wuji + wuji54`：

```python
import asyncio
import base64
import json
from pathlib import Path

import websockets

URI = "ws://172.31.0.226:8081"
OUT = Path("scene11_thirteen_random")
OUT.mkdir(exist_ok=True)


async def recv_type(ws, expected, timeout=180):
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if message.get("type") in expected:
            return message


async def main():
    async with websockets.connect(
        URI, max_size=64 * 1024 * 1024, ping_interval=None
    ) as ws:
        await ws.send(json.dumps({"type": "status"}))
        status = await recv_type(ws, {"status_response"}, 20)
        assert status["scene_id"] in {
            "showroom_scene_11", "showroom_scene_11_wuji"
        }, status

        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
            "showroom_scene_11_generalization": {
                "mode": "thirteen_random",
                "randomize_position": True,
                "randomize_yaw": True,
                "randomize_color": True,
                "randomize_table_color": True,
            },
        }))
        ack = await recv_type(ws, {"subscribe_step_result_response"})
        assert ack["ok"], ack
        assert ack["showroom_scene_11_generalization_mode"] == "thirteen_random"
        assert len(ack["showroom_scene_11_active_letters"]) == 13
        assert len(set(ack["showroom_scene_11_active_letters"])) == 13

        initial = await recv_type(ws, {"step_result"})
        frame = initial["frames"][0]
        (OUT / "initial.jpg").write_bytes(
            base64.b64decode(frame["image"]["data"])
        )
        (OUT / "layout.json").write_text(
            json.dumps(ack["showroom_scene_11_layout"], indent=2),
            encoding="utf-8",
        )


asyncio.run(main())
```

`single_custom` 的完整请求体示例：

```json
{
  "type": "reset_env",
  "showroom_scene_11_generalization": {
    "mode": "single_custom",
    "letter": "Z",
    "position": [0.10, -0.04],
    "cube_color": [0.80, 0.20, 0.20],
    "letter_color": [1.00, 1.00, 1.00]
  }
}
```

13 个指定字母示例：

```json
{
  "type": "reset_env",
  "showroom_scene_11_generalization": {
    "mode": "thirteen_custom",
    "letters": ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
    "randomize_position": true,
    "randomize_yaw": true,
    "randomize_color": false,
    "randomize_table_color": false
  }
}
```

上面的 `thirteen_custom` 示例会保留 client 指定的 13 个字母，但位置和 yaw 仍来自与训练一致的 tmp 布局槽位；关闭 `randomize_color` 和 `randomize_table_color` 后，积木与桌面恢复 authored 颜色，字体仍固定黑色。

`showroom_scene_11_wuji` 使用同一份 `scene_11.usd`，但机器人和 action contract 不同。从 O6 worker 进入时用 `switch_scene_and_layout` 原子切换；如果 worker 已经是 `wuji54`（例如当前运行 `pick_fruit_wuji`），也可以直接用 `switch_scene` 在两个 Wuji 场景之间切换。下面的命令是手工冷启动/恢复方式：

#### Scene11 机器人初始位姿

`showroom_scene_11` 和 `showroom_scene_11_wuji` 都支持在每次 rollout reset 时配置机器人 root 的初始位姿。字段为 `showroom_scene_11_robot_pose`；不传该字段时，server 使用当前 Scene11 默认位置和姿态。位置单位为米，姿态格式为 `quat_wxyz=[w, x, y, z]`。该字段可以和同一个请求中的 `showroom_scene_11_generalization` 一起使用。

位置有两种写法，**每个请求只能选一种**：

| 写法 | 含义 |
|---|---|
| `pos: [x, y, z]` | env-local 坐标系中的机器人 root 绝对位置；当前 Scene11 默认位置约为 `[0.003253, 1.395587, 0.76]` |
| `table_relative_pos: [x, y, z]` | 相对于当前 `qsim_table_1` 桌面 root 的位置；例如 `z=0.38` 表示机器人 root 比桌面 root 高 0.38m |

`pos` 与 `table_relative_pos` 同时传入会被拒绝，避免两种位置来源冲突；位置只能在这两种写法中选一种，但可以同时设置 `quat_wxyz`。`quat_wxyz` 可以单独传，此时位置仍使用默认位置；server 会归一化四元数。位置校验范围为绝对 env-local `x/y +/-3m`、`z 0.3~1.3m`。reset ack、subscribe ack、status 和初始及后续 `step_result` 的消息顶层 `showroom_scene_11_robot_pose` 会回显目标值与实际 PhysX root 位姿。

绝对位置和姿态示例：

```json
{
  "type": "reset_env",
  "showroom_scene_11_robot_pose": {
    "pos": [0.003253, 1.395587, 0.76],
    "quat_wxyz": [0.7071, 0.0, 0.0, 0.7071]
  }
}
```

相对桌面位置示例：standalone Scene11 当前桌面 root 约为 `[0.0, 1.8, 0.38]`，下面的请求对应机器人 env-local 位置约 `[0.003253, 1.395587, 0.76]`：

```json
{
  "type": "reset_env",
  "showroom_scene_11_robot_pose": {
    "table_relative_pos": [0.003253, -0.404413, 0.38]
  }
}
```

只设置初始姿态时：

```json
{
  "type": "reset_env",
  "showroom_scene_11_robot_pose": {
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
  }
}
```

该协议只对两个 Scene11 scene 生效，不改变水果、纸团/果皮、倒咖啡或其他 Showroom 场景。

```bash
PORT=8081 \
INITIAL_SCENE=showroom_scene_11_wuji \
ACTION_LAYOUT=wuji54 \
SCENE_STATE_FILE=/tmp/showroom_scene_11_wuji_scene_8081 \
ACTION_LAYOUT_STATE_FILE=/tmp/showroom_scene_11_wuji_layout_8081 \
PICK_FRUIT_LAYOUT_STATE_FILE=/tmp/showroom_scene_11_wuji_fruits_8081 \
  ./run_g1o6_server_loop.sh
```

连接后先检查：

```python
assert status["scene_id"] == "showroom_scene_11_wuji"
assert status["robot"] == "unitree_g1_wuji_bimanual_jointpos"
assert status["action_layout"] == "wuji54"
assert status["action_dim"] == 54
assert status["step_result_state_dim"] == 69
```

Wuji 场景的 hold action 也应按 `status.full_body_joint_names` 和 `status.action_joint_names` 动态抽取，不要读取硬编码的 `joint_position_51`；帧内通用字段 `state.joint_position` 可同时覆盖 O6/Wuji。

下面是 `showroom_scene_11_wuji` 的完整 client rollout 示例。示例假设 client 已通过原子协议或 operator 手工恢复让 worker 运行 `showroom_scene_11_wuji + wuji54`；它会做状态校验、强制 reset、保存初始帧、提交 3 个保持动作并保存动作后图片：

```python
import asyncio
import base64
import json
from pathlib import Path

import websockets

URI = "ws://172.31.0.226:8081"
OUT = Path("showroom_scene_11_wuji_rollout")
OUT.mkdir(exist_ok=True)


async def recv_type(ws, expected, timeout=180):
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") == "error":
            raise RuntimeError(msg)
        if msg.get("type") in expected:
            return msg


def save_frame(frame, path):
    image = frame["image"]
    path.write_bytes(base64.b64decode(image["data"]))


async def main():
    async with websockets.connect(
        URI,
        open_timeout=20,
        max_size=64 * 1024 * 1024,
        ping_interval=None,
    ) as ws:
        await ws.send(json.dumps({"type": "status"}))
        status = await recv_type(ws, {"status_response"}, 20)

        assert status["scene_id"] == "showroom_scene_11_wuji", status
        assert status["robot"] == "unitree_g1_wuji_bimanual_jointpos", status
        assert status["action_layout"] == "wuji54", status
        assert status["action_dim"] == 54, status
        assert status["step_result_state_dim"] == 69, status

        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
        }))
        ack = await recv_type(ws, {"subscribe_step_result_response"})
        assert ack["ok"], ack

        initial = await recv_type(ws, {"step_result"})
        assert initial["phase"] == "initial", initial
        initial_frame = initial["frames"][-1]
        save_frame(initial_frame, OUT / "initial.jpg")

        qpos69 = initial_frame["state"]["joint_position"]
        full_index = {
            name: index
            for index, name in enumerate(status["full_body_joint_names"])
        }
        action_names = (
            status.get("action_resolved_joint_names")
            or status["action_joint_names"]
        )
        hold_action = [float(qpos69[full_index[name]]) for name in action_names]
        assert len(hold_action) == status["action_dim"] == 54

        await ws.send(json.dumps({
            "type": "submit_actions",
            "actions": [hold_action] * 3,
        }))
        submit_ack = await recv_type(ws, {"submit_actions_response"})
        assert submit_ack["ok"], submit_ack

        post_actions = await recv_type(ws, {"step_result"})
        assert post_actions["phase"] == "post_actions", post_actions
        save_frame(post_actions["frames"][-1], OUT / "post_actions.jpg")


asyncio.run(main())
```

不要在一个 O6 worker 上依次发送 `switch_action_layout=wuji54` 和 `switch_scene=showroom_scene_11_wuji`。两次请求会分别重启并产生非法中间组合；跨机器人时发一条 `switch_scene_and_layout`。已经运行 `wuji54` 的 worker 可以直接在 `pick_fruit_wuji` 与 `showroom_scene_11_wuji` 之间调用 `switch_scene`。

### 4.6.3 laundryroom_r1_pro

R1 Pro 已有独立 client 手册，包含当前 `8080` 联调地址、四路相机、`r1pro32`、22D state、torso、机器人站位泛化和完整闭环代码：

- [`R1_PRO_CLIENT_USAGE_GUIDE.md`](R1_PRO_CLIENT_USAGE_GUIDE.md)

本通用手册只保留场景、协议和 layout 摘要；后续 R1 Pro client 专属内容统一更新到独立手册。

### 4.7 pour_coffee / pour_coffee_stereo rollout 泛化与双目示例

`pour_coffee` 和 `pour_coffee_stereo` 使用同一个 Area_9 场景：`Office_10F_Room01_pour_coffee.usd`。二者共享同一套 rollout 级泛化协议；区别是 `pour_coffee_stereo` 会额外输出左右目 RGB 图片。运行期泛化不走 hotkey；每个 `subscribe_step_result` 或 `reset_env` 默认按 `easim_ck` 同一套参数随机选择 source container 和 target cups，并随机桌面 XY / yaw。client 也可以用 custom 模式指定 kettle/cup 的桌面局部 XY。

当前资产池来自 `source/easim/scenario_loader/configs/pour_coffee_assets.yaml`：source container 为 `qsim_kettle_iron`，target cups 为 `qsim_white_cup`、`qsim_black_cup`。custom 模式只开放桌面局部 `xy` 和可选 `yaw_z_deg`；`z` 不由 client 指定，server 会保留 USD 里写好的高度。倒咖啡还支持 `robot_pose.pos_delta` 调整机器人相对默认站位的 offset，默认按 robot-local 坐标解释；在当前默认朝向下，`pos_delta[0] > 0` 表示朝桌子方向靠近，`pos_delta[1]` 表示机器人左右横移。

支持的模式：

| 模式 | 请求写法 | 行为 |
|---|---|---|
| 默认初始化 | `"mode": "default"` 或 `"mode": "default_visible"` | 使用 YAML `default_visible` 中的默认 kettle/cups，每个 rollout 仍随机桌面 XY 和 yaw |
| 随机泛化 | `"mode": "random"` 或 `"mode": "random_objects"` | 从可用 source container 和 target cups 池中随机选择对象，并随机桌面 XY 和 yaw；当前资产池只有 1 个 kettle + 2 个 cup，所以种类固定但位置/yaw 会泛化 |
| 固定对象 | `"mode": "fixed"` 或 `"mode": "fixed_objects"` | client 指定对象 leaf；server 固定这些对象种类，只随机桌面 XY 和 yaw |
| 自定义位置 | `"mode": "custom"` 或 `"mode": "custom_objects"` | client 指定 kettle/cups 的 leaf、桌面局部 `xy` 和可选 `yaw_z_deg`；server 校验 XY 在桌面采样范围内且对象不重叠 |
| 机器人站位 offset | `"robot_pose": {"pos_delta": [0.06, 0, 0]}` | 在当前默认机器人位姿基础上平移 root；默认 `frame="robot_local"`，`delta_x > 0` 表示朝桌子靠近，`delta_y` 表示左右横移；x/y 最大 `±0.20m`，z 最大 `±0.05m` |

切到双目倒咖啡场景：

```json
{
  "type": "switch_scene",
  "scene_id": "pour_coffee_stereo"
}
```

`switch_scene` 会触发 worker 重启；client 需要断开后重连，并用 `status` 确认：

```json
{
  "scene_id": "pour_coffee_stereo",
  "stereo_enabled": true,
  "stereo_camera_keys": {
    "left": "viewport_cam",
    "right": "viewport_cam_right"
  }
}
```

随机泛化并订阅首帧：

```json
{
  "type": "subscribe_step_result",
  "force_reset": true,
  "pour_coffee_generalization": {"mode": "random"}
}
```

默认初始化模式：

```json
{
  "type": "reset_env",
  "pour_coffee_generalization": {"mode": "default"}
}
```

固定对象种类、只随机位置和 yaw：

```json
{
  "type": "reset_env",
  "pour_coffee_generalization": {
    "mode": "fixed",
    "active_objects": {
      "source_container": "qsim_kettle_iron",
      "target_cups": ["qsim_white_cup", "qsim_black_cup"]
    }
  }
}
```

也可以把对象直接放在顶层字段，等价于 fixed 模式：

```json
{
  "type": "reset_env",
  "pour_coffee_source_container": "qsim_kettle_iron",
  "pour_coffee_target_cups": ["qsim_white_cup", "qsim_black_cup"]
}
```

自定义 kettle/cups 的桌面局部位置，并让机器人更靠近桌子：

```json
{
  "type": "reset_env",
  "pour_coffee_generalization": {
    "mode": "custom",
    "robot_pose": {
      "pos_delta": [0.06, 0.0, 0.0]
    },
    "source_container": {
      "leaf": "qsim_kettle_iron",
      "xy": [0.32, 0.00],
      "yaw_z_deg": 8.0
    },
    "target_cups": [
      {"leaf": "qsim_white_cup", "xy": [-0.12, -0.08], "yaw_z_deg": 0.0},
      {"leaf": "qsim_black_cup", "xy": [0.10, 0.08], "yaw_z_deg": -10.0}
    ]
  }
}
```

`xy` 是 Area_9 tabletop-local 坐标，单位是米。当前校验范围与随机采样参数一致：kettle `x=[0.15, 0.32], y=[-0.08, 0.08]`，cup `x=[-0.12, 0.10], y=[-0.08, 0.08]`。`robot_pose.pos_delta` 默认是 robot-local 坐标，单位也是米；`[0.06, 0, 0]` 会让机器人沿自身前向移动 6cm，在当前倒咖啡默认朝向下就是更靠近桌子。调试时可以把 offset 放大到 `[0.15, 0, 0]` 或 `[0.20, 0, 0]` 看明显效果；`0.20m` 是当前 x/y 允许上限。左右横移可用 `[0, 0.15, 0]` 或 `[0, -0.15, 0]`。若需要世界坐标 offset，可显式传 `"frame": "world"`。如果位置超范围或对象互相重叠，server 会在 `reset_env_response` / `subscribe_step_result_response` 中返回 `ok=false` 和错误原因。

几组常用机器人站位 offset 写法：

```json
{"robot_pose": {"pos_delta": [0.06, 0.0, 0.0]}}
{"robot_pose": {"pos_delta": [0.15, 0.0, 0.0]}}
{"robot_pose": {"pos_delta": [0.20, 0.0, 0.0]}}
{"robot_pose": {"pos_delta": [0.0, 0.15, 0.0]}}
{"robot_pose": {"pos_delta": [0.0, -0.15, 0.0]}}
```

reset / subscribe ack 会回显：

```json
"pour_coffee_generalization_mode": "default_visible" | "random_objects" | "fixed_objects" | "custom_objects",
"pour_coffee_active_objects": {"source_container": "...", "target_cups": ["...", "..."]},
"pour_coffee_layout": [...],
"pour_coffee_robot_pose": {"pos_delta": [0.06, 0.0, 0.0], "frame": "robot_local", "world_delta": [0.0, 0.06, 0.0], ...}
```

`pour_coffee_stereo` 的首帧或后续帧会返回左右目图片。推荐从 `frames[i].images` 读取；旧字段 `observation.image` 仍是左目，兼容单目 client：

```python
import base64

def save_stereo_frame(step_result, prefix):
    frame0 = step_result["frames"][0]
    left = frame0["images"]["left"]["data"]
    right = frame0["images"]["right"]["data"]
    open(f"{prefix}_left.jpg", "wb").write(base64.b64decode(left))
    open(f"{prefix}_right.jpg", "wb").write(base64.b64decode(right))
```

完整 client 示例：切到 `pour_coffee_stereo`，用 custom 布局开始一个 rollout，保存首帧左右目，并执行 5 次闭环 action。真实策略只需要替换 `policy()` 即可。

```python
import asyncio
import base64
import json
from pathlib import Path

import websockets

URI = "ws://172.31.0.226:8081"
SCENE_ID = "pour_coffee_stereo"
OUT_DIR = Path("pour_coffee_stereo_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_b64_image(image_obj, path: Path):
    path.write_bytes(base64.b64decode(image_obj["data"]))


async def recv_type(ws, expected_types, timeout=240):
    expected = set(expected_types)
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") in expected:
            return msg


async def get_status(uri: str) -> dict:
    async with websockets.connect(uri, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "status"}))
        return await recv_type(ws, {"status_response"}, timeout=30)


async def ensure_pour_coffee_stereo(uri: str) -> dict:
    status = await get_status(uri)
    if status.get("scene_id") == SCENE_ID and status.get("stereo_enabled") is True:
        return status

    async with websockets.connect(uri, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "switch_scene", "scene_id": SCENE_ID}))
        # switch_scene 会触发 worker 重启；不同版本可能直接断开，所以这里不依赖 ack。
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except Exception:
            pass

    for _ in range(80):
        await asyncio.sleep(3)
        try:
            status = await get_status(uri)
        except Exception:
            continue
        if status.get("scene_id") == SCENE_ID and status.get("stereo_enabled") is True:
            return status
    raise TimeoutError(f"worker did not switch to {SCENE_ID}")


def policy(step_result: dict, action_dim: int) -> list[list[float]]:
    # 替换成真实模型推理输出。server 期望 actions 形状为 [T, action_dim]。
    return [[0.0] * action_dim for _ in range(30)]


async def run_rollout():
    status = await ensure_pour_coffee_stereo(URI)
    action_dim = int(status["action_dim"])
    print("scene", status["scene_id"], "layout", status["action_layout"], "dim", action_dim)

    async with websockets.connect(URI, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
            "pour_coffee_generalization": {
                "mode": "custom",
                "robot_pose": {
                    # robot-local +X 是朝桌子方向；这里让机器人靠近 6cm。
                    "pos_delta": [0.06, 0.0, 0.0],
                },
                "source_container": {
                    "leaf": "qsim_kettle_iron",
                    "xy": [0.32, 0.00],
                    "yaw_z_deg": 8.0,
                },
                "target_cups": [
                    {"leaf": "qsim_white_cup", "xy": [-0.12, -0.08], "yaw_z_deg": 0.0},
                    {"leaf": "qsim_black_cup", "xy": [0.10, 0.08], "yaw_z_deg": -10.0},
                ],
            },
        }))

        ack = await recv_type(ws, {"subscribe_step_result_response"})
        assert ack.get("ok"), ack
        print("mode", ack.get("pour_coffee_generalization_mode"))
        print("active", ack.get("pour_coffee_active_objects"))
        print("layout", ack.get("pour_coffee_layout"))

        step_result = await recv_type(ws, {"step_result"})
        assert step_result.get("phase") == "initial", step_result.get("phase")

        frame0 = step_result["frames"][0]
        save_b64_image(frame0["images"]["left"], OUT_DIR / "initial_left.jpg")
        save_b64_image(frame0["images"]["right"], OUT_DIR / "initial_right.jpg")

        for step_idx in range(5):
            await ws.send(json.dumps({
                "type": "submit_actions",
                "actions": policy(step_result, action_dim),
            }))
            submit_ack = await recv_type(ws, {"submit_actions_response"})
            assert submit_ack.get("ok"), submit_ack
            step_result = await recv_type(ws, {"step_result"})
            last_frame = step_result["frames"][-1]
            save_b64_image(last_frame["images"]["left"], OUT_DIR / f"step{step_idx:02d}_left.jpg")
            save_b64_image(last_frame["images"]["right"], OUT_DIR / f"step{step_idx:02d}_right.jpg")
            print("step", step_idx, "phase", step_result.get("phase"), "executed", step_result.get("executed_actions"))


asyncio.run(run_rollout())
```

需要关闭 rollout 泛化时传 `"pour_coffee_generalize": false`。


### 4.8 pick_fruit server 端评测示例

当前评测模块放在 server 端。Client 只需要：

1. 用 `start_evaluation` 指定场景和指标，拿到 `evaluation_session_id`。
2. 每个 rollout 正常 `reset_env` / `subscribe_step_result`，额外带上 `evaluation_session_id`。
3. 算法动作结束后发 `finish_episode`，拿本次 `evaluation_report`。
4. 全部 rollout 结束后发 `finish_evaluation`，拿 server 汇总好的 `evaluation_summary`。

仿真平台按 `task_id` 聚合展示一次 client 测评任务。推荐一次测评任务只调用一次 `start_evaluation`，中间包含多个 rollout；如果确实需要跨多个连接或 worker 创建多个 evaluation session，也要给这些 `start_evaluation` 传同一个 `task_id`。不传 `task_id` 时，server 会用本次 run_id 作为默认 task_id，平台无法自动把多个 run 合并成同一个测评任务。

```python
import asyncio
import json
import websockets

URI = "ws://172.31.0.226:8080"

async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "start_evaluation",
            "scene_id": "pick_fruit",
            "evaluation": {
                "task": "objects_in_container",
                "metrics": ["pass", "pass_at_1", "pass_at_2", "pass_at_3"],
                "debug_details": False,
            },
        }))
        start = json.loads(await ws.recv())
        assert start["ok"], start
        sid = start["evaluation_session_id"]

        for _ in range(100):
            await ws.send(json.dumps({
                "type": "subscribe_step_result",
                "evaluation_session_id": sid,
                # 单水果："pick_fruit_single_fruit": "apple"
                # 固定三水果："pick_fruit_active_fruits": ["qsim_apple_2", "qsim_pumpkins", "qsim_carrot"]
                # 随机三水果：不写 pick_fruit_active_fruits / pick_fruit_single_fruit
            }))
            ack = json.loads(await ws.recv())       # subscribe_step_result_response
            initial = json.loads(await ws.recv())   # phase=initial step_result
            assert ack["ok"], ack
            assert initial["phase"] == "initial"

            # 在这里按现有闭环流程读取 step_result、提交 submit_actions，直到本次 rollout 结束。

            await ws.send(json.dumps({
                "type": "finish_episode",
                "evaluation_session_id": sid,
            }))
            report = json.loads(await ws.recv())
            assert report["ok"], report
            print(report["metrics"])

        await ws.send(json.dumps({
            "type": "finish_evaluation",
            "evaluation_session_id": sid,
        }))
        summary = json.loads(await ws.recv())
        assert summary["ok"], summary
        print(summary["pass_rates"])

asyncio.run(main())
```

输出要点：

- 单水果 rollout 的 `evaluation_report.metrics.pass` 表示唯一 active fruit 是否最终在碗内；`evaluation_summary.pass_rates.pass_rate` 是多 rollout 通过率。
- 多水果 rollout 的 `evaluation_report.metrics.pass_at_1/pass_at_2/pass_at_3` 分别表示至少 1/2/3 个 active fruits 最终在碗内；汇总字段是 `pass_at_1_rate/pass_at_2_rate/pass_at_3_rate`。
- `debug_details=true` 时会额外返回 `objects_in_container_count`、`objects_in_container`、`pose_source`、`object_xyz`、`container_xyz`、`thresholds` 和 `object_debug` 等中间量；默认推荐关闭，让 client 只消费最终指标。
- `pose_source="physx_world"` 表示评测读取的是 PhysX 运行时最终刚体位姿，适合判断视频里已经被移动过的水果；如果出现 fallback，会在 `pose_source` 里体现。
- `object_debug` 会给出每个 active fruit 的 `xy_distance`、`z_offset`、`in_xy`、`in_z` 和 `fail_reason`，用于排查“视频看起来成功但指标失败”的边界 case。

#### pick_paper_balls 评测指标和成功标准

`pick_paper_balls` 使用同一套 `objects_in_container` 评测协议，但最终成功指标只建议 client 消费 `metrics.pass` 和 summary 里的 `pass_rates.pass_rate`。

| 字段 | 含义 | 推荐用途 |
|---|---|---|
| `evaluation_report.metrics.pass` | 单个 rollout 是否成功 | 最终成功/失败判定 |
| `evaluation_summary.pass_rates.pass_rate` | 多 rollout 的成功率 | 平台或 client 展示总通过率 |
| `debug_details.objects_in_container_count` | 本次有多少 active 纸团/果皮进了任一可用垃圾桶 | 调试用 |
| `debug_details.container_pose_ok` | 所有 active 垃圾桶是否都保持直立 | 调试用；false 时本 rollout 必然 fail |
| `debug_details.containers_not_upright` | 被判定打翻的 active 垃圾桶 leaf 列表 | 调试用 |

`metrics.pass=true` 需要同时满足：

1. 场景里所有 active `orange_peels` 都进入任一 active `trash_cans`。
2. 场景里所有 active `paper_balls` 都进入任一 active `trash_cans`。
3. 所有 active `trash_cans` 都没有被打翻。

如果桌面上只有 1 个 active 垃圾桶，那么就是“所有果皮和纸团都进这个垃圾桶，并且这个垃圾桶没被打翻”才算成功。垃圾桶姿态用运行时 quaternion 判断，当前阈值是 local Z 轴相对世界 Z 轴倾斜不超过 45 度；超过阈值时 `container_pose_ok=false`，`metrics.pass=false`。

`pass_at_1/pass_at_2/pass_at_3` 对 `pick_paper_balls` 只是诊断型 object-count 指标，表示至少有几个 active 物体满足 in-container 条件；它们不是最终成功标准。正式评测推荐只传 `"metrics": ["pass"]`。

纸团果皮评测优先读取运行时 PhysX/world pose，而不是只看初始 USD layout；如果发生 fallback，会在 `pose_source` 中标记。`debug_details=true` 时重点看 `objects_in_container_count`、`objects_not_in_container`、`object_results`、`container_xyz_by_leaf`、`container_quat_wxyz_by_leaf`、`container_pose_ok`、`containers_not_upright`、`thresholds` 和每个 object 的 `object_debug`。

#### pick_paper_balls 单垃圾桶评测示例

下面示例创建一个 `pick_paper_balls` evaluation session，使用 custom layout 固定 1 个果皮、1 个纸团和 1 个垃圾桶；动作结束后调用 `finish_episode` 拿单 rollout 结果，再用 `finish_evaluation` 拿汇总 `pass_rate`。

```python
import asyncio
import json
import websockets

URI = "ws://172.31.0.226:8080"

async def main():
    async with websockets.connect(URI, max_size=32 * 1024 * 1024,
                                  open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "start_evaluation",
            "scene_id": "pick_paper_balls",
            "evaluation": {
                "task": "objects_in_container",
                "metrics": ["pass"],
                "debug_details": True,
            },
        }))
        start = json.loads(await ws.recv())
        assert start["ok"], start
        sid = start["evaluation_session_id"]

        await ws.send(json.dumps({
            "type": "reset_env",
            "evaluation_session_id": sid,
            "pick_paper_balls_generalization": {
                "mode": "custom",
                "orange_peels": [
                    {"leaf": "qsim_orange_peel", "xy": [0.18, 0.08]}
                ],
                "paper_balls": [
                    {"leaf": "qsim_paper_ball_3", "xy": [0.48, 0.18]}
                ],
                "trash_cans": [
                    {"leaf": "qsim_desk_trash_can_1", "xy": [0.30, 0.18]}
                ],
            },
        }))
        ack = json.loads(await ws.recv())
        assert ack["ok"], ack

        # 在这里按正常闭环流程发送动作，直到本次 rollout 结束。

        await ws.send(json.dumps({
            "type": "finish_episode",
            "evaluation_session_id": sid,
        }))
        report = json.loads(await ws.recv())
        assert report["ok"], report

        success = report["metrics"]["pass"]
        debug = report.get("debug_details", {})
        print("success:", success)
        print("objects_in_container_count:", debug.get("objects_in_container_count"))
        print("container_pose_ok:", debug.get("container_pose_ok"))
        print("containers_not_upright:", debug.get("containers_not_upright"))

        await ws.send(json.dumps({
            "type": "finish_evaluation",
            "evaluation_session_id": sid,
        }))
        summary = json.loads(await ws.recv())
        assert summary["ok"], summary
        print(summary["pass_rates"]["pass_rate"])

asyncio.run(main())
```

`reset_env` 里的 custom layout 只是在 rollout 开始时设置初始物体和垃圾桶；真正评测仍然在 `finish_episode` 时读取运行期 PhysX/world pose。因此算法动作把物体移进垃圾桶后，`metrics.pass` 会按最终状态计算。

默认：评测任务会为每个 rollout 落盘首帧和末帧 JPG，视频仍然默认关闭。client 用法和原来完全一致，不需要额外传 `record_artifacts`；如果需要完全关闭图片/视频落盘，可以在 `start_evaluation` 里传 `"record_artifacts": false`。需要保存视频或调整图片参数时，再传下面这种 dict 配置。

```json
{
  "type": "start_evaluation",
  "scene_id": "pick_fruit",
  "task_id": "platform_artifacts_demo",
  "task_name": "pick_fruit_artifacts_demo",
  "requested_rollouts": 1,
  "client": {
    "client_id": "demo_client",
    "source": "manual"
  },
  "evaluation": {
    "task": "objects_in_container",
    "metrics": ["pass_at_1", "pass_at_2", "pass_at_3"],
    "debug_details": false
  },
  "record_artifacts": {
    "initial_image": true,
    "final_image": true,
    "video": true,
    "max_video_frames": 60,
    "video_fps": 10,
    "jpeg_quality": 85
  }
}
```

`start_evaluation_response.platform_record.run_dir` 会返回本次任务目录；每次 `finish_episode` 的 `evaluation_report.artifacts` 会返回该 rollout 的相对路径。默认不传 `record_artifacts` 时通常只有 `initial_image` 和 `final_image`，显式打开 `video=true` 后才会多出视频字段，例如：

```json
{
  "initial_image": "artifacts/rollout_000001_initial.jpg",
  "final_image": "artifacts/rollout_000001_final.jpg",
  "video": "artifacts/rollout_000001.mp4",
  "video_frame_count": 60
}
```

完整文件路径可以用 `platform_record.run_dir + '/' + artifacts.video` 拼出来。8090 调试 worker 的一份已验证示例在：

```text
/kairos_vepfs_volc/simulation/xuqiang/easim_platform_data/ip-172-31-0-226/evaluation_runs/20260602_081255_eval_e713b406
```

如果 client 发现某次自动评测结果和视频不一致，推荐重跑同一个 rollout 配置并打开 `debug_details=true`，然后把以下两样信息一起提供给排查方：

1. 这次 `finish_episode` 返回的完整 `evaluation_report`，尤其是 `debug_details`。
2. 对应失败 rollout 的视频，或最后 3-5 帧能看到水果和碗的截图。

只要有这两项，通常就能判断是 PhysX pose 读取、XY 半径、Z 高度阈值，还是视频视角造成的边界误解。水果种类和模式不需要额外口述，`evaluation_report.mode`、`active_objects`、`object_set_key` 已经包含这些信息。`evaluation_summary` 可选；单次误判最关键的是那一次的 `evaluation_report.debug_details`。


### 4.9 调试：每个 rollout 自动记录 torso / camera pose

如果想排查闭环过程中画面轻微抖动，可以让 server 在每个 rollout 自动记录 `torso_link` 和胸前 `viewport_cam` 的 pose。Client 不需要判断什么时候成功，也不需要在成功后手动触发；只要打开 debug 开关，server 会从该 rollout 的 initial frame 开始记录，后续每个 `env.step` 都写一条 JSONL。

打开方式：在 `subscribe_step_result` 或 `reset_env` 里带上：

```json
{
  "debug_rollout_pose_log": true
}
```

同一条连接内，开关打开后会持续生效：后续每次 `subscribe_step_result` / `reset_env` 开始新 rollout 时，server 会自动关闭上一份 log 并新建一份 log。传 `"debug_rollout_pose_log": false` 或 client 断开连接时，server 会关闭当前 log。

最小示例：

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8090"  # 推荐只在调试 worker 上用

async def main():
    async with websockets.connect(URI, max_size=64 * 1024 * 1024,
                                  open_timeout=30, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
            "pick_fruit_single_fruit": "apple",
            "debug_rollout_pose_log": True,
        }))
        ack = json.loads(await ws.recv())      # subscribe_step_result_response
        initial = json.loads(await ws.recv())  # phase=initial step_result
        assert ack["ok"], ack
        print("pose jsonl:", ack["rollout_pose_debug_path"])
        print("summary:", ack["rollout_pose_debug_summary_path"])

        # 后续按原闭环流程 submit_actions。server 会自动在每个 env.step 写 pose log。
        # 下一次 rollout 再发 subscribe_step_result / reset_env，即可自动生成新的 log 文件。

        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "force_reset": True,
            "pick_fruit_single_fruit": "apple",
        }))
        ack2 = json.loads(await ws.recv())
        initial2 = json.loads(await ws.recv())
        assert ack2["ok"], ack2
        print("next rollout pose jsonl:", ack2["rollout_pose_debug_path"])

asyncio.run(main())
```

server 端输出位置类似：

```text
/root/code/easim/server_logs/rollout_pose_debug_8090_rollout_pose_20260528_103012/
  rollout_000001_torso_camera_pose.jsonl
  rollout_000001_summary.json
```

`rollout_*.jsonl` 每行包含：`frame_kind`、`latest_step`、`executed_actions`、`external_action`、`torso_link.pose_w`、`camera.pose_w`、以及相对第一帧和上一帧的 `deltas`。`summary.json` 汇总最大位移和最大角度变化。

判断思路：如果 torso 和 camera 同步变化，视角抖动大概率来自机器人机身/上身微动；如果 torso 稳定但 camera 变化，优先查 camera sensor / attachment；如果二者都稳定但图像仍抖，优先查渲染、图像编码或 client 显示链路。

---

## 5. 常用 one-liner

```bash
# 探针 4 worker 当前状态
python3 -c "
import asyncio, json, websockets
async def m():
  for p in (8080,8081,8082,8083):
    try:
      async with websockets.connect(f'ws://172.31.0.226:{p}', open_timeout=3) as ws:
        await ws.send(json.dumps({'type':'status'}))
        r = json.loads(await asyncio.wait_for(ws.recv(),timeout=3))
        print(p, r.get('scene_id'), r.get('action_layout'), r.get('action_dim'),
              'busy' if r.get('ws_client_count',0) > 0 else 'idle')
    except Exception as e:
      print(p, 'ERR', type(e).__name__, e)
asyncio.run(m())
"

# 把 worker 8083 切到 pick_fruit + arm26（独占，需要确认无人在用）
python3 - <<'PY'
import asyncio, sys
sys.path.insert(0, 'source/easim/scripts')
from g1o6_action_layout import request_action_layout_switch
asyncio.run(request_action_layout_switch('ws://172.31.0.226:8083', 'arm26'))
PY
python3 -c "
import asyncio, json, websockets
async def m():
  async with websockets.connect('ws://172.31.0.226:8083') as ws:
    await ws.send(json.dumps({'type':'switch_scene','scene_id':'pick_fruit'}))
    print(await ws.recv())
asyncio.run(m())
"
# 然后等 ~150-200s pick_fruit 冷启动完成

# 原子把 8081 从 O6 切到 Wuji 水果场景，并等待完整 contract 就绪
/workspace/isaaclab/_isaac_sim/python.sh \
  source/easim/scripts/_switch_scene_and_layout.py \
  --uri ws://172.31.0.226:8081 \
  --scene_id pick_fruit_wuji \
  --action_layout wuji54

# 原子把一个空闲 worker 切到 R1 Pro 洗衣场景
/workspace/isaaclab/_isaac_sim/python.sh \
  source/easim/scripts/_switch_scene_and_layout.py \
  --uri ws://172.31.0.226:8081 \
  --scene_id laundryroom_r1_pro \
  --action_layout r1pro32

# 跑一次 pool smoke（4 worker 健康度 + protocol 自检）
python3 _smoke_test_pool.py
```

---

## 6. 排查速查

| 现象 | 通常原因 + 处理 |
|---|---|
| `ConnectionRefusedError` 或 `open_timeout` | worker 没在 LISTEN：可能正在切场景/layout 自重启（等 60–180s），或 pool 整体挂了 → 见 [`RESTART_GUIDE.md`](RESTART_GUIDE.md) |
| `submit_actions_response` `ok=false`，error 含 `dim=X expected=Y (action_layout=...)` | client 用错 dim：cfg 是 36 维但 worker 是 arm26，或反之 → `--switch_action_layout` 切，或修 cfg |
| `step_result` `frames` 为空 / 接收超时 | 没订阅就 submit；server 不会推 step_result，只回 submit_actions_response。要 push 必须先 `subscribe_step_result` |
| `unknown message type: switch_action_layout` | server 是旧版（v2，不支持 switch_action_layout）。整池重启走 [`RESTART_GUIDE.md`](RESTART_GUIDE.md) §1.1，或对该 worker 触发 `switch_scene` 让它 wrapper-restart 加载新版 server |
| `unknown message type: switch_scene_and_layout` | 当前 worker 仍是旧进程；先完整重启该 worker加载新版 server。不要退回串行发送 scene/layout 两条切换请求 |
| `Use switch_scene_and_layout to change scene_id and action_layout atomically` | 单独 `switch_scene` 或 `switch_action_layout` 会产生非法跨机器人 pair；改发一条原子请求 |
| `scene_id='laundryroom_r1_pro' requires action_layout='r1pro32'` | R1 Pro scene/layout 没有成对切换；使用 `switch_scene_and_layout` 原子请求 |
| R1 Pro 四路图像缺失或 `image_camera_keys` 不完整 | 确认启动时带 `--enable_cameras`，完整重启 worker，并检查相机 prim 是否挂在独立 `/Robot/root_joint/...` 下 |
| `pick_fruit scene unavailable on this deployment` | sim host 缺 `easim.scenario_loader.configs.pick_fruits_room01_scene` 或 Office_10F_Room01 / fruit USD 资产 → 检查 `source/easim/scenario_loader/configs/pick_fruits_room01_scene.py` 和 `assets/` |
| `pick_fruit reroll failed; requested fruit pool is unavailable` | 固定指定的 leaf 在当前 USD stage 里没有 prim；100 个 fruit leaf 已配置在 YAML，但当前 Office USD 可能只包含部分资产 → 换成可解析 leaf，或补齐对应 USD prim |
| `stack_plate_bowl runtime layout failed` | 当前 USD stage 里无法解析 2 个 plate + 1 个 bowl，或请求的 leaf 不存在 → 检查 `stack_plate_bowl_assets.yaml`、Office Room01 Area_6 USD 和请求字段 |
| `unknown stack_plate_bowl mode` | `mode` 只能用 `default/default_visible`、`fixed/fixed_objects`、`random/random_objects` |
| `pick_paper_balls layout reroll failed` | 当前 USD stage 里无法解析 1 个橘皮 + 1 个纸团 + 1 个垃圾桶，或请求的 leaf 不存在 → 检查 `pick_paper_balls_assets.yaml`、Office Room01 Area_7 USD 和请求字段 |
| `unknown pick_paper_balls mode` | `mode` 只能用 `default/default_visible`、`fixed/fixed_objects`、`random/random_objects`、`custom/custom_objects`；固定模式必须同时给 `orange_peel` / `paper_ball` / `trash_can`，custom 模式同类 leaf 不能重复且数量不能超过资产池 |
| `unknown pick_fruits_and_paper_balls mode` | `mode` 只能用 `default/default_visible`、`random/random_objects`；当前还不支持 `fixed` / `custom` |
| `unknown scene_id 'pick_fruit'` / `unknown scene_id 'stack_plate_bowl'` / `unknown scene_id 'pick_paper_balls'` / `unknown scene_id 'pick_fruits_and_paper_balls'` / `unknown scene_id 'pour_coffee'` | server 是更老版本（连 `_VALID_SCENE_IDS` 都没更新）→ 整池重启加载磁盘最新版 |
| `scene_switching` push 没收到、连接直接关 | 正常：server `os._exit(50)` 太快可能在 push 到达前已断；客户端只需 reconnect-poll status 直到 scene 变成目标值即可 |
| 多 client 同时连同一端口 → 结果错乱 | 池协议不强制阻止；调用方自己保证独占。可用 `status_response.ws_client_count` 探活 |

---

## 7. 相关文档导航

| 文档 / 文件 | 内容 |
|---|---|
| 本文 ([`CLIENT_USAGE_GUIDE.md`](CLIENT_USAGE_GUIDE.md)) | G1/Wuji 客户端使用速查 + 各场景 rollout 泛化和评测（最新入口）|
| [`R1_PRO_CLIENT_USAGE_GUIDE.md`](R1_PRO_CLIENT_USAGE_GUIDE.md) | R1 Pro 四相机 MIL、torso、站位泛化与完整 client（R1 Pro 专用入口） |
| [`SCENE_SWITCH_CLIENT_GUIDE.md`](SCENE_SWITCH_CLIENT_GUIDE.md) | `switch_scene` / `switch_scene_and_layout` / `switch_pick_fruit_layout` 协议，以及机器人-layout 配对和 rollout 泛化边界 |
| [`ACTION_LAYOUT_GUIDE.md`](ACTION_LAYOUT_GUIDE.md) | full36 / arm26 / wuji54 / r1pro32，51D/69D/22D state 顺序、action→state 映射和跨机器人原子切换 |
| [`RESTART_GUIDE.md`](RESTART_GUIDE.md) | 整池/单 worker 重启、`ACTION_LAYOUT` / `INITIAL_SCENE` env 配置 |
| [`FEATURE_MIGRATION_RUNBOOK.md`](FEATURE_MIGRATION_RUNBOOK.md) | `easim_ck -> easim` 新功能移植流程、资产/USD 检查、worker 重启和 smoke checklist |
| [`source/easim/scripts/g1o6_action_layout.py`](source/easim/scripts/g1o6_action_layout.py) | layout 常量库 + `request_action_layout_switch` / `request_scene_and_layout_switch` / `request_pick_fruit_layout_switch` async helper |
| [`source/easim/scripts/_switch_scene_and_layout.py`](source/easim/scripts/_switch_scene_and_layout.py) | O6/Wuji/R1 Pro scene + action layout 原子切换 CLI |
| [`source/easim/scripts/_switch_pick_fruit_layout.py`](source/easim/scripts/_switch_pick_fruit_layout.py) | pick_fruit 启动期 layout 切换 CLI |
