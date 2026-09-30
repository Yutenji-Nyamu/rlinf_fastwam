# SZ3 Wan Goal 环境验收

2026-09-30 23:33。两套官方安装已正常退出，尚未借卡或执行GPU smoke。

## 已验证

- 固定三套模型共36.78GB全部下载；765个运行文件大小、7个大文件SHA256、496个普通初态和246个KIR初态均通过。真实目标tokenizer的官方SHA256也通过。
- OFT环境7项、π05环境12项实际导入均通过，涵盖Torch、torchvision、FlashAttention、WanBackend和对应模型loader。
- π05实际输入变换、归一化、腕图mask、actor replay一致性、H10预测/C8执行以及smoke/formal配置检查通过。尚未加载GPU模型，不能据此声称GRPO已更新。

共同版本：Python3.11.14、Torch2.11.0+cu130、torchvision0.26.0+cu130、FlashAttention2.8.3、Ray2.58.0、DiffSynth1.1.9、DeepSpeed0.18.4、protobuf6.33.6。OFT实际transformers4.40.1（`bc339d9…`），OpenVLA-OFT源`e4287e9…`。π05实际导入OpenPI的transformers4.53.2，stock distribution元数据仍显示4.57.6；tokenizers0.21.4，JAX0.5.3、Orbax0.11.13。

## 依赖报告的处理

`uv pip check`不是全绿：OFT保留6条、π05保留5条元数据警告，原始退出码1保留。

|项目|依据与处理|
|---|---|
|旧OpenVLA/OpenPI/LeRobot声明的Torch系列版本|固定RLinf `pyproject.toml`110–123行明确覆盖为Torch2.11、torchvision0.26、torchcodec0.11、protobuf6.33.5以上；保留官方覆盖|
|π05 stock transformers与fork的tokenizers要求冲突|固定`install.sh`2416–2421行明确说明两个distribution共用目录，并要求tokenizers<0.22；实际fork及模型loader导入通过|
|OFT TensorFlow2.15与protobuf6|RLinf为Ray显式选protobuf6；TensorFlow导入打印`GetPrototype`诊断但OFT loader返回成功。记录此边界，GPU smoke仍须通过|
|OFT tyro/typeguard、SwanLab/wrapt|TensorFlow Addons/TensorFlow约束旧版本；本次入口为Hydra、日志为TensorBoard。保留安装结果，不为未使用的CLI和日志后端改训练依赖|
|π05 MuJoCo真实约束冲突|OpenPI附带的ALOHA/dm-control把MuJoCo升至3.8.1，超过rlinf-libero要求的<3.4。只把独立π05 venv的dm-control/MuJoCo对齐到1.0.34/3.3.7（固定官方installer也使用的配对）；现所有已安装包对这两项的有效约束均满足，Torch等未变|

`audit_environment_metadata.py`只接受上述已核对的具体警告，未知警告会失败。其`REVIEWED_FOR_GPU_SMOKE`表示可进入实际smoke，**不是pip检查全绿或GPU验收通过**。完整包清单保存在服务器`logs/*pip-freeze*.txt`；公开材料只保留必要版本与轻量回执。

## 证据

- `s109`：π05导入及接口检查通过；首次完整验证因pip警告退出1。
- `s110`：资产与tokenizer验收，退出0。
- `s114`：OFT实际导入通过；pip警告保留。
- `s115–s116`：追踪MuJoCo依赖来源、对齐两包、核全部有效约束，退出0。
- `s117/s119`：分别生成OFT/π05版本与已审警告回执。审计脚本初版误要求π05安装未使用的torchaudio、一次调用遗漏uv的PATH，均只修正审计入口，没有为此给模型环境增加包。

官方依据：[固定依赖覆盖](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/pyproject.toml#L110)、[OpenPI安装处理](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/requirements/install.sh#L2398)、[MuJoCo配对参照](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/requirements/install.sh#L3725)。
