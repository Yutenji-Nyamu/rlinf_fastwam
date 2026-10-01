# 深圳3 Wan Goal 继续实施 · 2026-10-01

用户起床后要求继续昨晚工作并检查三机。08:43实查12条原RLT都存活并继续推进：SZ1约629–642轮，SZ2约1349–1409轮，SZ3约126–132轮。SZ1、2保持原训练；SZ3在准备完成后借物理4–7卡继续Wan Goal单视角π0.5。0–3无新的任务安排。

## 10:04：正式首轮真实训练进展与两窗协调

`w059`确认r5正式第0轮完成：grad norm=0.7964912、loss mask=0.01259766、优势[-1.6201816,0.5400605]、有限loss0.00014402；首轮1061.4秒，已进入第1轮采集。WM奖励估计success=1.5625%，不等同于真实LIBERO成功率。四卡显存约62–62.5GiB；当前owner/直接RLT归还guard都存活，最近无primary error。正式首个CP仍在原定第40轮，本次没有改预算或存储间隔。

`w060/w061`再刷SZ1/SZ2：原四RLT均活并推进，分别655/649/661/657和1418/1444/1383/1385轮。用户要求两个exp窗口协调；“0927 exp”确认当前EXPO-FT只研究、GPU需求为0，不启停本窗口运行。分配与后续登记规则见[资源表](../server-admin/EXPERIMENT_RESOURCE_COORDINATION_20261001.md)。

## 09:35：π0.5已证实一轮真实更新，准备原预算正式训练

**09:51用户更改归还路线：WM结束或失败后直接恢复原四RLT，不再续Dojo评测。`w053`已在服务器唯一部署`post-wm-direct-rlt-v5`归还请求/CPU guard，绑定当前v5 owner，正式训练不受信号影响。旧frozen owner的兼容方式：等WM完整释放后中断其短暂Dojo启动入口，再由同一owner完成释放核查和唯一RLT归还；没有第二个RLT restorer。当前训练持续，旧WM→Dojo→RLT记录是历史。**

`w034`：r4的原定2轮smoke正常退出。第一轮32条轨迹全失败，官方奖励过滤令loss mask=0、grad=0，空样本优势统计为NaN；第二轮loss mask=0.23046875、grad norm=3.7244246、优势范围[-0.5400605,1.6201816]，WM估计成功2/32。后者是WM奖励，不是实际LIBERO成功率。完整CP1/CP2与16个trainable张量抽样权重变化已通过，最大差约3.8e-6。

原`smoke-verification.json.ok=false`如实保留：它要求两轮都有效，而本次只有一轮有效。用户要求的是验证真实有效参数更新；`w038`独立核第二轮真实GRPO信号、正常退出、完整存储、实际权重变化，并逐一检查两份checkpoint所有浮点参数均有限。新增`one-update-reconciled.json`明确只验收一轮有效更新，不声称两轮有效。该证据与原始回执、event文件、resolved配置、源补丁及释放回执以SHA绑定。

不扩smoke，不关闭官方过滤、不改seed/任务。正式仍从固定原SFT启动、1000轮，每40轮保存，N64/G8/R8/L320/C8、global2048/micro128、仅4–7。`w040`09:35:33仅TERM当前v4 outer，由其完成旧资源归还；`w043`新v5 prepare通过，`w044`09:40:42唯一launch成功。`w048`真实actor/env/rollout物理placement4–7通过；`w051`resolved正式配置全部复核与原YAML SHA一致。正式处于首轮8次rollout采集，尚无runner训练指标。禁止重放stop、prepare或launch。

`w028`缓存修复和轻量证据已发布，远端SHA `11e39d8b4b4bd76ed80ec9fc80a985d3d5e58a1d`，9新增/5修改/0删除。上述学习验收和formal-only路由为新增工作，尚待增量发布。

## 配方和验收

沿用RLinf `d34d4c320d08cb982de034aa9a011f08dc0fa217`及昨晚固定模型和独立环境；不重下载。OFT两次真实更新已由独立证据验收，原退出阶段monitor故障单独保留；本次复用`learning-reconciled.json`，不重跑OFT。

|阶段|环境N / GRPO组G / rollout轮R|最长步数L / 执行动作块C|global / micro batch|预算与存储|
|---|---|---|---|---|
|π0.5 smoke|32 / 8 / 1|320 / 8|1280 / 64|2轮，每轮完整checkpoint|
|π0.5正式|64 / 8 / 8|320 / 8|2048 / 128|1000轮，每40轮存储；从原SFT开始|

两阶段均为用户已批准的外部主相机RGB＋腕图mask，原动作维度与GRPO语义不变。原strict verifier要求两轮都有效，最新用户目标按已证实的一轮真实更新与完整CP/正常退出验收，见上节；不覆盖原失败回执。WM内部奖励与真实LIBERO成功率分别记录。当前启动入口`scripts/prepare_formal_resume.py --launch`已唯一执行；实际子入口`private_ray_driver.py --config wan_goal_pi05_headonly_formal_sz3`，独立Ray63845。

当前正式输出`/data/chenyiteng/projects/wan-goal-sz3/runs/wan-goal-sz3-20261001-r5`；模型、checkpoint与日志在数据盘，Inductor/Triton编译缓存使用各阶段独立tmpfs。仅GPU4–7，offload及actor/env/rollout placement沿原配置。WM结束或失败后按精确owned身份清理并直接续原四RLT，总预算仍3000轮；不续Dojo。共享Ray、其他用户和无关任务保持原状。以下v3/v4交接属于历史，禁止重放。

## 新资源交接

昨晚v2 owner已结束，但用户另行恢复了RLT，所以其历史`pipeline-final.json.rlt_dispatched=false`仍保留。当前权威恢复证据是v2目录`user-rlt-return.json`与旧cycle `resumed-dispatched.json`，08:43又实查四条真实进展。新prepare仅补齐该精确外部归还回执路径；仍核cycle、continuation、释放文件SHA、实际driver/namespace/完整checkpoint，不复用旧stop或restore。

新bridge为`P/scripts/wm-bridge-20261001-v3`，新cycle为`P/rlt-cycle-sz3-wan-goal-20261001-wake-v1`，新continuation为`D/continuation-20261001-wan-goal-v3`。P是SZ3 Dojo项目，D是原6300回合目录。新owner负责WM→原Dojo→原RLT的唯一归还链。`w006`准备成功并保留2313个Dojo回合，四条原RLT绑定最新完整CP125；`w007`于08:55唯一启动owner。禁止重放prepare/launch/stop。

## 现场证据

本地细日志`local_logs/wan-goal-20261001/steps/`：w001–w003三机只读快照；w004核旧owner已结束、外部RLT归还存在、13份运行源码SHA仍匹配、环境已审通过、63843/63844/63845未占用。SZ3根盘余5.57GB，数据盘余7.22TB；缓存/输出继续放数据盘。后续只追加真实执行证据。

08:57 `w011`确认原RLT释放后π05 smoke已启动，独立Ray63844；actor/env/rollout实际placement均物理4–7。模型正在加载，尚无真实更新验收。

09:02 `w015–w017`：首轮32环境rollout完成（79.36秒），actor训练前向在`_apply_rope`的TorchInductor加载生成`.py`时FileNotFoundError。未产生已验有效更新或checkpoint。Ray dashboard报错为该失败触发的退出处理次生错误；独立owned清理仍成功，随后按原owner续Dojo。09:04报错文件已能读到，底层是`fuse.mergerfs`；临时文件可见性是工作假设，尚未证明根因。

缓存处理沿[PyTorch官方环境变量](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)，保持编译及方法参数，仅把新run各阶段Inductor/Triton缓存放独立0700的`/dev/shm/chenyiteng-wan-goal-<run hash>/`。内存盘可用约1.06TB、挂载无noexec；模型、日志和checkpoint仍在数据盘。`w019`精确TERM当前v3 owner，令其负责Dojo清理与RLT归还，不直接杀Ray或workers；后续新owner借卡前核旧终态。

09:09双进程实际CPU编译/导入/前后向梯度匹配检查通过，CUDA未初始化；90.76秒，44缓存文件/0.75MB。09:10旧v3 owner已结束、归还派发成功、error=None。09:14 `w022`准备新bridge v4、continuation v4、cycle `rlt-cycle-sz3-wan-goal-20261001-wake-v2`、WM run r4；训练YAML SHA保持原样。09:15请求唯一启动，禁止重放准备/停止/启动。

用户已要求清理定时任务：本聊天统一为原ID `rlt`的“三机训练与资源归还检查”，每15min，正常推进静默；当前服务器唯一owner及绑定guard按最新WM→RLT请求归还。旧且暂停的9月17日π05准备检查已删除，旧Wan专用不重建；“每日调研”独立保留。
