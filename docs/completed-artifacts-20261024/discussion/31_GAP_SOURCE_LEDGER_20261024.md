# NFE、执行长度与蒸馏曲线：来源记录

核对日期2026-10-24。来源结果不能直接等同于我们的Sidney权重与仿真协议。

|来源|核实内容|本项目可用之处与边界|
|---|---|---|
|[Coda / Fewer Steps, Better Actions](https://arxiv.org/html/2609.21216)，§4.1、表I、附录A|50 RoboTwin任务、Clean50数据上另训π0.5 60k更新；执行C10；每任务50有效回合。原始N1 59.76%、N10 71.52%；N5 71.64%|最接近我们的模型和环境。支持扫描C10，但不能把其11.76个百分点差全归因C，因为权重、训练与协议也不同。多步收益也非一直单调|
|[SnapFlow](https://arxiv.org/html/2604.05656)，附录I|LIBERO-10，固定预测块，扫描执行1/3/5/10/20；教师N10与蒸馏学生N1的领先关系随C改变|支持执行长度敏感性分析，不是原始N1对N10的直接证据；不是RoboTwin，不能照抄最优C|
|[DriftOPD](https://arxiv.org/html/2610.00317)，表2|π0.5-LoRA RoboTwin、每任务30回合；按平均轨迹长<150、150–279、≥280分18/21/11任务。教师N10和原始N1：短60.74/61.30，中67.30/56.35，长55.15/40.61|步数收益可集中在中长任务。这里分组是轨迹长度，不是给任务硬设400动作上限；老师权重不是我们的Sidney|
|[OpenPI模型源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)|Euler积分dt=-1/N，完整走噪声到动作；每步用速度更新|N1不是仅执行N10第一小步。已对照本地实际调用链，N确实传入且没有最低5步截断|
|[OpenPI action chunk broker](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/action_chunk_broker.py)|执行配置的action_horizon后重新索取动作|预测长度和执行长度可不同，有明确官方接口依据；仍须维持我们的原生仿真控制|
|[OneDP ICML2025](https://proceedings.mlr.press/v267/wang25ba.html)|预训练扩散策略的一步分布蒸馏，强调较少蒸馏训练开销|方法原版是扩散策略；早期稿DDIM一步崩溃不能推断π0.5一步也必差|
|[Consistency Policy](https://arxiv.org/html/2405.07503v2)|EDM一致性/监督训练，初始噪声尺度也做过消融|噪声会影响策略，但不同模型训练约定不同；不直接在π0.5上复制EDM噪声参数|

主要建议来自Coda的近邻协议、本项目实测配对结果与已验收评估接口；SnapFlow/DriftOPD用于说明C和任务结构为何值得单独分析。未发现可证明“越短C必然扩大N10优势”或“统一400动作最公平”的依据。
