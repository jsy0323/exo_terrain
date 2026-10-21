# 局部软组织与护具接触（RL 简化版）

实现依据是 [rehyb_mujoco 的人体—外骨骼模型](https://github.com/stesfazgi/rehyb_mujoco/blob/main/models/exo_with_patient/nesm_with_patient.xacro)
及 [Tesfazgi et al., 2023](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2023.1167604/full)。
原实现用 composite 组织微元及网格绑带。这里保留其“骨骼—软组织—护具”承力关系，
将组织微元自由度凝聚成稀疏软接触，适合先做并行 RL 实验。
这是工程近似；当前参数经过数值与承力检查，尚未用实际护具的实验数据标定。

## 已修改的模型

| XML | 接触区域 | 现有踝部电机状态 |
| --- | --- | --- |
| `MFG_Raj2015_V11_randomterrainexo.xml` | 骨盆、左右大腿、左右小腿 | 保留两路拉索电机 |
| `MFG_Raj2015_V11_fixed_slope_6deg_exo.xml` | 同上 | 保留两路拉索电机 |
| `MFG_Raj2015_V11_fixed_stairs_exo.xml` | 同上 | 保留原文件的电机禁用设置 |

原 MyoSuite 人体关节、肌肉、肌腱和惯性参数保留。原有小腿刚性护具替换为接触护具；
新增骨盆／大腿框架后总质量比原随机地形外骨骼模型增加约 1.065 kg。
`passiveexo.xml`、`idealexo.xml` 和无外骨骼模型没有更改。

## 承力关系和简化

- 骨盆护具相对人体有 3 平移 + 3 转动自由度，没有回中弹簧、阻尼或刚性焊接。
- 骨盆护具承载独立的双侧外骨骼髋关节，每侧 3 个转动自由度，连接大腿护具。
  外骨骼髋中心相对人体髋中心偏移 `[5, -3, 0]` mm（人体局部坐标）。
  髋关节目前无电机、无助力弹簧，仅有 0.05 N·m·s/rad 的轴阻尼。
  因此骨盆／大腿部分用于局部耦合和被动框架受力，不会新增髋部 RL 动作。
- 每个小腿护具有 6 个相对自由度；拉索近端 `exo_cable_guide_r/l` 位于该护具上。
  助力经“电机→拉索→小腿护具→软接触／摩擦→胫骨”传递。
  远端足部端点仍代表与足部固定的鞋附件。
- 每处用前、后、内、外四个方向 × 两排接触垫，共 40 个显式接触对。
  人体侧球形 patch 固定在骨骼上，组织的等效压缩／回弹／阻尼由软接触求解器表示。
  不引入组织粒子自由度、体积守恒或组织内部剪切网络。
- 护具形状为分段固定环。2 mm 装配压缩量表示收紧后绑带的等效预紧；
  不模拟带体的伸长、扣具或独立的绑带张力状态。
- 所有 patch 的动态碰撞掩码关闭，只通过显式 pair 发生接触，避免与地面、肌肉绕行体、
  另一条腿和视觉网格产生额外碰撞。`load_mesh=false` 仍会保留接触 patch。

新增 24 个被动自由度，`nq/nv` 从 `53/52` 变为 `77/76`。人体侧自由度没有删除。

## 调参入口

修改 `tools/add_soft_contact.py` 的参数和 `pads()`，随后运行：

```powershell
python Env/MFG_Musculoskelet_V11/tools/add_soft_contact.py
```

脚本只替换带标记的区域和原小腿刚性护具，可重复运行；地形资源和原电机启用状态保留。
地形生成脚本也会从更新后的随机地形 XML 继承接触结构。

| 参数 | 初始值 | 含义 |
| --- | --- | --- |
| `PRECOMPRESSION` | 0.002 m | 接触垫装配压缩量，等效绑带收紧程度 |
| `CONTACT_TIME` | 0.01 s | 接触恢复时间常数；较大时更软 |
| `solref` 第二项 | 1 | 临界阻尼比 |
| `solimp` | `0.95 0.99 0.004 0.5 2` | 压缩相关的约束阻抗 |
| `FRICTION` | 0.7 | 两个切向方向的摩擦系数 |
| `HIP_OFFSET` | `[0.005, -0.003, 0]` m | 外骨骼髋轴中心错位 |
| `timestep` | 0.001 s | 保留原仿真步长及 implicitfast 积分器 |

这些 solver 参数不是直接的 N/m、N·s/m 或杨氏模量；实际等效刚度还随惯性、
姿态、接触数与阻抗变化，见 [MuJoCo 约束参数说明](https://mujoco.readthedocs.io/en/3.3.2/modeling.html#solver-parameters)。
需要按实际护具的压缩—力曲线、应力松弛和滑移实验重新拟合。
默认独立小腿测试台的八点法向力之和约 666 N；这是各接触点的标量和，
不是护具净作用力，也不是测得的绑带张力。
当前 patch 只能给出离散接触力；不能将球面点接触力直接当作连续皮肤压力分布。

## RL 接入

`mfgenv/human_state.py` 将人体参考状态映射到新增被动关节后的数组地址。
初始化时外骨骼髋角／角速度跟随参考人体髋关节；此后自由演化。
模仿奖励和人体 jerk 项只读取人体关节。
参考姿态预览、参考 site 运动学和轨迹回放也使用相同映射。

采用非对称 actor–critic 观测：

| 接收方 | 当前默认维度 | 接口状态 |
| --- | --- | --- |
| 人体 actor | 1069 | 保留原有观测顺序，不增加接口关节状态 |
| exo actor（`first256: True`） | 256 | 保留原公共观测的前 256 维，不增加接口关节状态 |
| critic | 1117 | 原 1069 维公共观测之后追加 48 维接口状态 |

新增的 48 维按 XML 中接口关节顺序排列：24 个位置／角度，随后是对应的 24 个速度。
其中包含骨盆相对运动、双侧外骨骼髋关节、双侧小腿护具相对运动。
Gym 的 `observation_space` 和 reset/step 返回的 `obs` 始终是公共 actor 观测。
额外状态通过 `info['critic_extra_obs']` 传递给向量环境包装器，由包装器拼接成
`extras['observations']['critic']`。PPO 单独存储、归一化并使用这一路状态估计价值；
两个 actor 的前向计算和推理接口只读取公共观测。

SB3 自动 reset 后，critic 的接口状态取自对应环境的 reset info，
避免将上一 episode 的终止状态配到新 episode 的 actor 观测上。
原有 AMP 关节索引与 `first256` 的输入含义保持不变。
物理接触对人体运动的影响仍会体现在原有的人体运动学观测中。

骨盆／小腿的保护行程是平移 ±30 mm、转动 ±0.25 rad。
`info['interface_travel_fraction']` 给出最大相对行程占比；达到 0.8 时
`terminated_info['interface_slip']` 触发 episode 终止，避免长期依靠行程限位传力。
髋关节范围是独立的机械关节活动范围。

默认训练环境的 actor 观测保持 1069 维，critic 增加到 1117 维，动作维度保持 108。
原有公共观测布局的 `ActorCriticTerrainExo` checkpoint 可以继续用于初始化：
两个 actor 和 actor 归一化统计保留；critic 首层追加的 48 列权重初始化为零。
critic 归一化扩展后重新累计统计；网络形状变化时跳过旧优化器状态。
也可从新策略开始：

```powershell
python trainV11_reload.py resume_path=null exo_checkpoint_path=null log_dir=logs/exo_soft_contact
```

训练 YAML 中原有 checkpoint 设置没有自动覆盖。楼梯可视化所用模型仍按其原有配置禁用电机。
上一版将接口关节交错放入公共观测的 1117 维 checkpoint 不适用于当前公共观测布局。
尚未运行完整步行 PPO 训练，不能据短时检查推断策略已收敛。

## 验证

依赖按项目 `requirements.txt`：MuJoCo 3.3.2、Gymnasium 1.1.1。

```powershell
python -m unittest discover -s tests -p test_soft_contact.py -v
python -m pytest tests/test_privileged_observations.py tests/test_actor_critic_terrainexo.py -q
```

覆盖三个 XML 的肌肉／位置控制、开关视觉网格的 12 种加载组合；人体状态映射；
参考轨迹 reset、模仿奖励与环境 step；200 N 轴向加 15 N 横向载荷；
偏心 200 N 拉索；三个真实参考步态姿态下各 1.5 s 的支撑试验。
独立测试台移除行程限位，确保受力平衡来自接触；关闭接触后护具自由加速。
这些是受控数值测试，完整步行仍需要重新训练和评估。
另覆盖 DummyVecEnv／SubprocVecEnv 的自动 reset、两个 actor 的输入隔离、
独立 actor／critic 归一化、一次小网络 PPO 更新、checkpoint 扩展及可视化维度推断。

本机一次 CPU 支撑姿态微基准（关闭视觉网格、2000 个物理步）中，原刚性模型约
36.8 μs/步，当前模型约 76.0 μs/步，物理步开销约为原来的 2.1 倍。
该测试不包含 PPO、环境观测、地形行走或多进程通信，不能据此直接推算训练吞吐量。
