# SZ-H100 Git remote、worktree 与实现总流水（2026-08-22）

边界：本账从用户确认 deploy key 已在 GitHub 登记并授权实现/推送后开始。普通项目操作使用
`chenyiteng`；不记录密码、private key或token，不force-push。GRPO、π0 telemetry与Fast-WAM telemetry
各自另有分账；本账只记共同preflight、Git认证、remote和worktree。

## IMP-000 — 17:31：现场与精确目标 preflight

状态：`PASS`。

- 本地命令文件：`local_scripts/remote_commands/shenzhen_implementation_preflight_20260822.sh`
- SHA-256：`86C8AF39FA2423875AD69AE27D0F15F93BE21B01D734A96D0CBEF655573B5E44`
- 执行入口：固定host-key校验的`verified_password_ssh.py`，用户`chenyiteng`；密码只进入当前交互进程。
- 身份：`uid=1003(chenyiteng)`，组含`sudo`、`labdata`。
- PPO现场：driver存活；已完成`Global Step 38/100`，正在下一步rollout；`fatal_count=0`；checkpoint为10/20/30。
- 资源：GPU 0约69 MiB，1--3约4 MiB，4--7约60.6--61.4 GiB；driver cgroup
  `memory.current=1,895,754,833,920 B`（约1.724 TiB），整机`MemAvailable=280,336,932 kB`
  （约267.3 GiB），无cgroup OOM事件。
- RLinf：canonical HEAD精确为`7d07a4212ee6858cc333e1d4fab7a37256d1f839`；GRPO与π0 telemetry
  目标branch/path均不存在。
- Fast-WAM：canonical HEAD精确为`7faa71108368fbb3b6885649f112af607427a2d4`；只有既有
  `third_party/RoboTwin/policy/fastwam_policy`符号链接为untracked；telemetry目标branch/path不存在。
- SSH key：private/public mode分别为600/644，指纹
  `SHA256:rHlM9Y7x89Y+1n4F98ZYB70oHiTiQuPbw2YU8oBGs24`，与用户截图一致；专用known-hosts尚未建立。
- 磁盘：`/`剩236 GiB，`/home`剩2.2 TiB，`/data`剩3.1 TiB。

判断：代码、Git与worktree操作可继续；PPO内存需等Step 40同相位刷新后再判断是否安全延长到Step 60。

## IMP-010 — 17:32：GitHub认证与`personal` remote

状态：`PASS`。

- 本地命令文件：`local_scripts/remote_commands/shenzhen_github_personal_remote_activate_20260822.sh`
- SHA-256：`022AF8C046B6D04A18BE042F485A4D0D46A22450D4F4C14F40B1B683A630FFF6`
- 写入仓库专用known-hosts并验证GitHub官方ED25519 fingerprint：
  `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`。
- 用deploy key执行只读认证探针：`git ls-remote git@github.com:Yutenji-Nyamu/rlinf_fastwam.git HEAD refs/heads/main`
  成功，远端HEAD/main均为`8138d6700e3838250c1139289ebfba43d48ff7de`。
- 在canonical RLinf仓库新增`personal=git@github.com:Yutenji-Nyamu/rlinf_fastwam.git`，保留
  `origin=https://github.com/RLinf/RLinf.git`。
- 设置repo-local `core.sshCommand`，固定专用private key、专用known-hosts与
  `StrictHostKeyChecking=yes`；不影响该用户其他Git仓库。
- `git fetch --prune personal`成功，发现main及DSRL/DVAC/OGPO/QAM/RLT等历史分支。

判断：截图中的read/write deploy key已经实际完成Git认证；RLinf隔离分支后续可以直接push到用户仓库。

## IMP-020 — 17:33：三个隔离worktree

状态：`PASS`。

- 本地命令文件：`local_scripts/remote_commands/shenzhen_create_isolated_worktrees_20260822.sh`
- SHA-256：`5182F4B60DAB7094E437BB557697012F348E25D7CCF72A3FBDA8D01BFBD8322E`
- GRPO：branch `codex/sz-7d07a421-grpo-pi0-robotwin`，worktree
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421`，起点`7d07a421...`。
- π0 telemetry：branch `codex/sz-current-pi0-dvac-observe`，worktree
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421`，起点`7d07a421...`。
- Fast-WAM telemetry：branch `codex/sz-fastwam-dvac-observe`，worktree
  `/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711`，起点`7faa711...`。
- 三个新worktree创建后均为clean；canonical与既有PPO worktree均未改动。

判断：三条实现线已做到代码与历史隔离，可并行开发；Fast-WAM仍需独立GitHub fork/repo后才能获得正确的push目标。

## IMP-030 — 18:18：Fast-WAM独立deploy key预备

状态：`PASS`；只生成服务器侧专用key，尚未认证或push。

- Fast-WAM与`rlinf_fastwam`是不同Git历史；GitHub deploy key按仓库隔离，不能复用前一把repo key。
- 命令文件：`local_scripts/remote_commands/shenzhen_fastwam_deploy_key_prepare_20260822.sh`；SHA-256：
  `4A5DAE4B7FDFDA4C1DB0402632608A79F710FA1F9903E76DD023E1344834F2BF`。
- 新key路径：`/home/chenyiteng/.ssh/github_fastwam_deploy_ed25519{,.pub}`；mode为600/644，owner均为
  `chenyiteng`。private key内容没有输出或落入本地文档。
- 公钥：

  ```text
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINeBC6bmHwe95L9NV6MF1QBOwymDhuGVWPSdafs406Qr SZ-H100 FastWAM
  ```

- 指纹：`SHA256:AluK6VvoK6VcCUVhlILEh6F77cimXdywhQkLOUt6BVE`。
- 下一步需要用户先在GitHub把official `yuantianyuan01/FastWAM` fork为
  `Yutenji-Nyamu/FastWAM`，再在新fork的Settings → Deploy keys登记上述公钥并勾选写权限。完成后才会
  建立Fast-WAM repo-local SSH remote并普通push；不会把Fast-WAM提交塞入RLinf仓库。

## IMP-040 — 三条实现线发布结果汇总

状态：RLinf两线已push；Fast-WAM已local commit、待独立fork登记key。

| 实现线 | branch | commit | 发布状态 |
|---|---|---|---|
| current π0 RoboTwin GRPO | `codex/sz-7d07a421-grpo-pi0-robotwin` | `554c6dc8d586162d9444c01fa88308ed4f5203d0` | `personal`普通push，remote/upstream一致，clean |
| current π0 DVAC observe | `codex/sz-current-pi0-dvac-observe` | `800baf80d6eab64169cf0e691eb04a681a093ee9` | telemetry parent=`f7cf0f60...`；parity gate已`personal`普通push，remote/upstream一致，0/0，clean |
| official Fast-WAM DVAC observe | `codex/sz-fastwam-dvac-observe` | `c63dc9b5384d6637a93cc862dbe2815d0332801d` | telemetry parent=`fc652fb49...`；parity gate exact local commit，clean；未push |

- π0 parity发布脚本SHA-256=`eb1d6aa49da40ad27dec0d78485f546135070d3f292f31ca2301682d1757840d`；
  exact1 staged file `359/0`、unstaged0后提交，无force push后local/remote/upstream全部等于
  `800baf80...`，ahead/behind 0/0。
- Fast-WAM parity local-commit脚本SHA-256=
  `1a67219407c763e7c59ba3af883ffe64fdd5948821058dfe8b7597eaf942e58a`；exact3 staged files
  `384/1`、unstaged0后提交为`c63dc9b5...`，parent=`fc652fb49...`，clean，push=0。
- GRPO实际只新增一份176行YAML；current compose/GRPO advantage探针通过。详见12号流水。
- π0 DVAC exact6 telemetry files + 1 real-parity tool、Fast-WAM DVAC exact7 telemetry files + 3 real-parity
  files均完成focused tests与独立review，无阻塞项；π0 parity commit=`800baf80...`已无force普通
  push，Fast-WAM parity commit=`c63dc9b5...`只在本地提交。两侧均未运行真实GPU/model/simulator。
  详见13号流水与Fast-WAM专题DVAC流水。
- 真实GRPO smoke与两侧DVAC P0均保留在resolved packet批准边界；代码完成/CPU检查不冒充真实runner或
  simulator验证。
