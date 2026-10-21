根据以下文章 建模跟精细exoskeleton与人体contact:
Tesfazgi et al., 2023 
Uncertainty-aware automated assessment of the arm impedance with upper-limb exoskeletons
论文全文 · 开源代码：rehyb_mujoco 
2. Mele et al., 2025 — 直接针对绑带与压力分布
A Novel Elastic Model for Exoskeleton-User Coupling Interfaces，ICORR 2025
Wang et al., 2025 — 说明接触方向会改变助力传递
Human-Interface Dynamics of Knee Exoskeletons with Lateral and Anteroposterior Attachment，ICORR 2025
Zhang et al., 2021 — 分布式接触点的参考
Modeling and Simulation of a Human Knee Exoskeleton’s Assistive Strategies and Interaction

保留现有 MyoSuite 肌骨模型。 
给外骨骼与人体保留必要的相对运动。 如果现有实现把电机力矩直接加到人体关节，或将两者完全刚性固定，需要检查新增接触是否真正承担助力传递。
先精细化主要承力部位。 从实际护具覆盖的骨盆／大腿区域开始，加入软组织等效刚度、阻尼、摩擦、绑带预紧和关节错位。
先参考 rehyb_mujoco 做局部软组织与护具接触