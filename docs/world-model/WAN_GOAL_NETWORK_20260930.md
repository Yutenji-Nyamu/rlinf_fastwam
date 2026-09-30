# SZ3 Wan Goal 下载线路记录

更新时间：2026-09-30 22:58。目标是获取同一组官方固定版本，线路切换不改变权重或训练配置，依赖继续遵循官方版本约束并记录最终freeze。

## 本机现状

已读取服务器 `/README_to_codex.md` 和 `/home/readme_network_to_codex.md`。默认 shell 继承团队代理 `127.0.0.1:7890`；另一现有代理 `127.0.0.1:7897` 也在运行。所有切换只作用于本任务进程，没有更改共享代理节点、系统服务或其他用户配置。

判断直连要显式使用 `curl --noproxy '*'` 或关闭客户端环境代理继承。早期探针只有“DIRECT”标签、未清除环境变量，不能当作真实直连证据。

## 本次实际比较

| 对象与线路 | 结果 | 采取措施 |
|---|---|---|
| Hugging Face 官方 Wan 权重，7890 | 同一 8 MiB 片段 35 秒超时，只传约 2.8 MiB，约 82 KiB/s | 从本任务下载进程撤下 |
| 同一文件、同一区间，7897 | HTTP 206，8 MiB / 2.34 秒，约 3.4 MiB/s | 21:45 断点续传切至此线路 |
| 7897 持续下载 | 21:50 两个文件分别约 3.0、3.4 MiB/s；Wan 约 1.3 GiB、OFT 首片约 1.1 GiB | 维持 2 文件 × 8 连接，继续观察 |
| HF Xet + 7890 | 出现 403、进度停滞 | 使用官方 resolve URL + aria2，保留固定 SHA 校验 |
| hf-mirror 真直连 | 1 MiB 小样本较快；16 MiB 样本 25 秒仅约 2.23 MB | 不因短样本结果替换持续下载线路 |
| PyTorch R2 + 7890 | 两套官方 installer 多次 TLS EOF 后失败 | 单独获取同一 wheel，再续官方安装 |
| PyTorch 主站真直连 | 曾短时有吞吐，持续 aria2 后 TLS 失败；21:51 8 MiB 探针 35 秒只传 15.5 KiB | 保留已下载约 201 MiB 的有效分块，改走 7897 |
| PyTorch 主站／R2 + 7897 | 同一 8 MiB 区间均约 29 秒完整返回；实际8连接续传均速约1.9MiB/s | 21:52 主站 + 7897 断点续传，21:55下载与SHA校验完成 |

短样本用于选路，最终以持续吞吐、完成回执和校验为准。模型下载和依赖安装各有独立日志，网络失败不意味着 GPU 算法失败；此时 Dojo 仍在运行，未借卡。

## 不改版本的下载方式

模型均来自 `WAN_GOAL_OFFICIAL_RUNBOOK.md` 中的固定官方仓库 revision。大文件用 aria2 断点续传与官方 LFS SHA256 校验，完成后再移到模型目录；小文件仍用 Hugging Face Hub。

PyTorch 是官方安装器当前选择的 `torch-2.11.0+cu130-cp311-cp311-manylinux_2_28_x86_64.whl`。主站与 R2 域名不同，但验收仍要求锁文件中的同一 SHA256：

```text
225b22e0a4e36ea573d3a68796e6816a160616f67e8b8c55683a88bf7777f4cd
```

文件大小531,045,934字节。21:55 aria2与独立Python SHA256都通过。随后只将这个wheel用于两套本实验独立环境，安装方式与结果另记运行日志。未安装到现有 Dojo、RLT 或公共环境。

复用方式是 `uv --no-config pip install --python <目标venv>/bin/python --no-index --find-links <wheel目录> --no-deps 'torch==2.11.0+cu130'`；先清本进程 `UV_TORCH_BACKEND`，按包名从本地目录安装。这样保留registry类型元数据，官方下一次 `uv sync` 可复用同版本；直接把wheel路径作为包参数可能触发来源类型不一致而重装。依据：[uv0.12.5 flat index](https://github.com/astral-sh/uv/blob/0.12.5/crates/uv-resolver/src/flat_index.rs#L115)、[同步判断](https://github.com/astral-sh/uv/blob/0.12.5/crates/uv-installer/src/satisfies.rs#L65)。预装后检查精确version以及无 `direct_url.json`，再原样运行官方installer。最终环境版本以全部安装后的freeze为准，不把中途Torch版本预先当成最终环境锁。

## 可用但未采用的后备

- ModelScope 官方 `Wan-AI/Wan2.2-TI2V-5B` 的 VAE 与本次清单大小、SHA256 一致；本次 VAE 已完整取得，无需重复下载。尚未找到可核验的三套完整任务权重官方镜像，不能用通用 Wan DiT 替代 LIBERO Goal DiT。
- 历史记录有“深圳2下载 → 内网转深圳3”的成功路径。本轮尚未实测深圳2，因此只作为下一种可验证的路线，不沿用旧速度作当前结论。

## 22:42–22:46 环境依赖的另一处瓶颈

`s085`：OFT下载完LIBERO素材后，ManiSkill Git fetch发生TLS EOF；上游安装器正执行最多5次重试。π05已下载完LIBERO素材，CUDA/JAX大包仍下载并重复重试。这些与三套模型权重下载分属不同进程。

`s086–s087` 用同一个 cuDNN 9.27.0.42 x86_64 wheel、同一8MiB区间比较：PyPI真直连0字节超时；7890约2.6KB/s；7897约258KB/s；腾讯PyPI镜像真直连3,063,699B/s、2.738秒完整HTTP206。镜像index中的SHA256与官方PyPI一致；清华约19.7KB/s，阿里和华为该次index没列此精确wheel。**这只是选路样本，尚未更换正在运行的安装进程，也未宣称持续大文件完成速度。** 可按同版本同SHA单独下载并复用，最终版本仍以官方解析实际结果为准。

ManiSkill同源码的后备传输已核：[官方tag API](https://api.github.com/repos/haosulab/ManiSkill/git/ref/tags/v3.0.0b22) 将v3.0.0b22指向 `33967b9e3ead1f841eec57cc9f31d0d8b8cf0907`；[固定源码包](https://codeload.github.com/mani-skill/ManiSkill/tar.gz/33967b9e3ead1f841eec57cc9f31d0d8b8cf0907)可用，setup.py静态版本3.0.0b22、无子模块。若重试仍失败，可保留原安装器其余参数，仅将此Git来源替为同commit归档；单独预装archive后又运行原GitURL仍可能重复拉Git，因此需明确记录这一传输替换。此方案尚未实施。

## 证据

22:54已仅给π05安装进程设置腾讯PyPI默认index及该域名NO_PROXY，精确停旧安装器后重跑原官方安装。OFT随后靠上游原有Git重试成功构建同一ManiSkill提交，因此22:57取消备用归档下载，未生成或采用install-network.sh。该取消不属于待修复失败。详见s089–s097及运行日志。

细粒度回执：当前聊天 E 盘 `wan-goal-20260930/steps/`。`s063–s064` 读本机网络指引；`s065` 同区间代理比较；`s066` 精确停止本任务 aria2 后按 7897 续传；`s067` 持续吞吐；`s068–s069` 安装失败与 PyTorch 线路对比；`s070` PyTorch 续传。

每步保留命令、标准输出／错误、时间、退出码和身份校验。共享代理秘钥与用户账户密码不写入文档或仓库。
