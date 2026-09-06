# OIDN 开关小尝试账本

2026-09-04。本轮用户明确授权一些尝试：聚焦显式开/关OIDN、各推理两三次，不将原生生命周期诊断或严格多臂实验作为前置。

## 授权与范围

- 在chenyiteng账号的独立诊断进程和输出目录准备/运行小规模渲染、推理；运行前展示具体配置、命令、资源和停止条件。
- 旧Python生命周期补丁先保留；不同时回撤，不升级库，不更改共享site-packages，不训练、不续训，不碰Sidney/shared Ray/其他用户。
- 各开关先各3个短回合，不将该数量作为效果证明；遇到实际故障只作窄修，不扩大实验。

## 操作记录

1. 完整读取根入口、窗口交接与Fast-WAM唯一current SSOT；读取既有固定host-key SSH helper。
2. 2026-09-04 11:48 CST：普通账号固定host-key、进程内密码只读身份探针成功，uid=1003/chenyiteng。未借用管理员权限。
3. 搜索RLinf/RoboTwin官方issue与SAPIEN源码，核对显式none入口；新线索RoboTwin #477报告关OIDN后仍可发生其他渲染hang，不能承诺全面根治。
4. 下一操作：只读刷新GPU/RAM、两项run、checkpoint、代码HEAD/dirty，并定位当前模型小推理入口与实际denoiser参数路径。
5. 11:50现场：GPU6/7各5MiB、无compute进程；RAM available=1,272,171,810,816B；Sidney Step40、所查fatal/OOM均0；Fast最后Step33、exit255。RLinf HEAD=4faade1d50bf、RoboTwin补丁HEAD=8c7380c118ce，均clean。
6. 首次只读脚本最后的rg目录枚举失败（服务器无rg），其前面的资源/代码读取成功；改用find。两个猜测YAML路径不存在，随后从实际文件和runtime/resolved.yaml获取配置。未据这些错误推断模型不可用。
7. 后续find核验Step30 DCP两分片各约14.45GB、metadata约2.92MB，均存在；未作DCP恢复。小对照计划用原release SFT checkpoint，经现行RLinf Fast-WAM policy推理，不将其冒充Step30模型。
8. 新确认参数链：RLinf cfg.task_config→VectorEnv args→task.setup_demo kwargs→Base_Task；Base_Task内部setup_scene()不传kwargs。因此只改setup_scene中的kwargs.get不生效，最小接法为_init_task_env_保存self.ray_tracing_denoiser，再在setup_scene使用。
9. 开始隔离实现：新分支codex/sz-robotwin-oidn-toggle、新worktree robotwin-oidn-toggle-20260904，基于旧补丁8c7380c1。仅增加一个默认oidn的参数和替换一处硬编码；不修改原工作树、vector_env或任何C++库。
10. git apply首次因无上下文hunk被拒（无半应用），补上准确上下文后成功，diff-check通过；准备脚本一次猜测pretrained_models目录不存在，去掉无关ls探针后继续，未安装/下载任何资产。
11. 新worktree三个资产目录只创建指向已有只读资产的symlink；试验脚本与输出只在该worktree/diagnostics。试验具体范围定为开/关各3短回合、各最多192动作步/8次C24策略query，0更新；使用同3个eval成功seed池条目、SFT release模型，保留旧补丁。
12. prepare导出整份official配置时遇到与本次推理无关的EVALUATION.output_dir Hydra resolver缺失；窄修为只完整导出实际使用的model/processor及device/dtype覆盖，不改模型配置，不导出未使用的官方评估器字段。GPU推理尚未启动。
13. 12:06—12:11，两模式各3回合正常完成，图像/动作与日志已采集。图库生成发生fontconfig缓存不可写警告，但PNG成功生成并已逐图检查，未为此改系统缓存权限。
14. 收尾发布检查阻止向origin官方仓库推送，改为读取并确认既有personal用户fork；没有修改remote。首次commit因该RoboTwin仓库未配置身份而停止，未提交；复用Fast-WAM仓库已配置的用户身份，仅对本次git命令用-c传入，未持久化作者身份。
15. 12:20 CST：仅envs/_base_task.py提交为`b76d4edcb7e23f09f9de6dafc16a3173f6d07adb`，推送personal的`codex/sz-robotwin-oidn-toggle`成功并建立新分支tracking；未向官方origin推送。诊断脚本oidn_toggle_trial.py仍是该独立worktree中的本轮未跟踪文件，完整本地副本已保存，不将该worktree宣称全clean。原RoboTwin修复树和RLinf树仍clean。
16. 收尾现场：GPU6/7各5MiB，原gcs_server/raylet/Sidey driver PID 321933/322685/3176215均仍在；未停止或重启。静态简图与完整图已打开核验。

## 结果：开关已可用，关闭后不是零影响

2026-09-04 **12:06:26—12:11:04 CST**，GPU6独立执行，OIDN/none两进程均exit0；未检出OIDN Error、pthread_key_create、Python fatal、OOM或Traceback。共6回合、47次策略query（on23/off24）、提交1128个C24动作槽，0优化器更新；提交动作槽不等于实际物理仿真step。原生库没有升级，没有创建Ray job。

| seed | OIDN开 | OIDN关 | 初始模型输入MAE（0—255） | 首块关节动作平均绝对差（rad） | 最大绝对差（rad） |
|---|---|---|---:|---:|---:|
| 100100001 | 未成功，8 query | 未成功，8 query | 2.489 | 0.01162 | 0.05636 |
| 100100005 | 未成功，8 query | 未成功，8 query | 2.483 | 0.00769 | 0.04347 |
| 100100018 | 成功，7 query | 未成功，8 query | 2.454 | 0.00567 | 0.03829 |

- 两侧初始14D状态逐值一致、prompt一致；从配对图可见物体与相机布局匹配。未保存/逐值验证全部物体pose，所以不宣称严格物理状态全等；未做on/on重复渲染，像素差也包含渲染随机性。
- 视觉判断：场景结构、物体和pad颜色保留，但无OIDN在桌面、阴影、腕部画面有明显颗粒。缩放后模型输入MAE约2.45—2.49/255；原始三相机约3.50—3.77/255。平均差小不等于局部边缘无变化。
- 首块12个关节通道平均变化0.0057—0.0116rad（约0.33—0.67°），最大0.038—0.056rad（约2.2—3.2°）；夹爪通道平均绝对差约0.0019—0.0041（原命令单位）。后续闭环会积累差异，不能仅凭肉眼判断策略无感。
- 本小样本成功1/3对0/3：既不能证明关闭损害成功率，也不能宣称观测无影响；两种模式短程都不崩，不能据此确认长程key故障已根治。
- 模型是**原始release SFT**，不是Step30 RL checkpoint；此处不回答Step30模型的收益或恢复可用性。
- 12:08采样GPU6约28,409MiB、该进程RSS约8.6GiB；这是采样值，不是GPU/RAM全程峰值。结束后GPU6/7均5MiB，原Sidney GPU4/5仍使用，shared Ray未干预。

图：[简图](oidn-toggle-20260904/comparison-brief.png)、[三例完整图](oidn-toggle-20260904/comparison.png)、[可拖动查看过程的页面](oidn-toggle-20260904/index.html)、[指标JSON](oidn-toggle-20260904/comparison.json)。仅下载8,494,508B小证据；checkpoint/模型留在服务器。

## 具体怎么关

[RLinf #947原评论](https://github.com/RLinf/RLinf/issues/947#issuecomment-4153077553)确实只建议将`set_ray_tracing_denoiser("oidn")`改成`set_ray_tracing_denoiser("none")`；**主动关闭不需要先修C++错误处理或追完整资源所有权**。另一份[RoboTwin #477](https://github.com/RoboTwin-Platform/RoboTwin/issues/477)报告Blackwell上关闭后仍可渲染hang，属于其他硬件/栈的风险线索，不是本H100必然复现的结论。

本次独立RoboTwin分支`codex/sz-robotwin-oidn-toggle`只改`envs/_base_task.py`两处，已推送[提交b76d4ed](https://github.com/Yutenji-Nyamu/RoboTwin/commit/b76d4edcb7e23f09f9de6dafc16a3173f6d07adb)：

```python
# _init_task_env_，setup_scene之前：
self.ray_tracing_denoiser = kwags.get("ray_tracing_denoiser", "oidn")
# setup_scene，创建相机之前：
sapien.render.set_ray_tracing_denoiser(self.ray_tracing_denoiser)
```

选用这个独立副本（`ROBOTWIN_PATH`及`PYTHONPATH`包含该worktree），未来完整RLinf运行可传：

```text
+env.train.task_config.ray_tracing_denoiser=none
+env.eval.task_config.ray_tracing_denoiser=none
```

两个`+`分别是两行各自的Hydra新增字段前缀，不是`++`语法。只关train不关eval，会保留eval中的OIDN。未选用新代码时，旧BaseTask仍写死oidn，单传该参数不会生效。默认仍为oidn，当前没有把正式训练改成none。

审计记录：每个进程构造/重置时的passthrough setter日志均与选择一致（on四次oidn，off四次none），BaseTask实例字段也一致；没有为了通过测试替换setter值。源码设置位于相机创建前，与原生显式none路径一致。

## 旧补丁怎么处理与下一步

关闭OIDN**不会使旧Python补丁自动失效**：vector_env的reset线程、child关闭及global cache清理逻辑仍执行，作用范围不止OIDN。第一轮保留它有明确意义：不同时改变第二个因素。本轮未回撤，其文件hash与原修复工作树一致。

建议现在把none视为已能运行的备选，而不是“已证明等价的修复”。若用户愿意接受颗粒与小幅动作变化，可以继续明确标记的无OIDN短程验证；无需先开展C++追踪。若选择它作为后续运行方式，再单独审计旧补丁保留/撤除，不将所有历史修改永久留存。不自动启动长程GRPO或改Sidney。

## 启动配置与命令

配置prepare已通过：seed=100100001/100100005/100100018，旧补丁保留，B1，rt/spp32/depth8/H32/C24/M10，eval_seed=0。完整配置见[OIDN on](oidn-toggle-20260904/oidn-resolved.yaml)、[off](oidn-toggle-20260904/none-resolved.yaml)、[实际模型/processor](oidn-toggle-20260904/official-resolved.yaml)。

启动采用local_scripts/remote_commands/sz_oidn_toggle_run_20260904.sh：普通账号独立进程，physical GPU6、各3短回合、0更新、各模式20分钟保险超时、明确异常停止本次尝试；不创建Ray job。预计GPU约20—40GiB、CPU RAM数十GiB（估计，不是已测峰值）。输出/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/oidn-toggle-20260904/{oidn,none}。复用已有DIFFSYNTH模型路径和ModelScope cache，不安装或下载模型。
