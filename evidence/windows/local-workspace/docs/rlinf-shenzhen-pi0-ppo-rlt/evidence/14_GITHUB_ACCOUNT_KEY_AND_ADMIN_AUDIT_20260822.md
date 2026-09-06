# SZ-H100 GitHub 账号级 SSH key 与管理员只读巡检（2026-08-22）

机器：`SZ-H100` / `admin` / `120.241.223.9:22`。连接全程使用固定 host-key
`SHA256:qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY` 的 Paramiko 密码路线；密码只进入当前进程，
没有写入命令文件或账本。普通 key 操作使用 `chenyiteng`，全机元数据巡检使用 `toom` 的只读 sudo。

边界：用户明确要求把原先单仓 deploy-key 路线扩展为常见的 GitHub **账号级 Git SSH key**。本轮只生成
不覆盖现有文件的新 keypair并交付公钥；没有修改任何 Git remote、SSH config、known-hosts，没有认证
GitHub、创建 fork 或 push。管理员巡检不读取其他用户文件正文、shell history或凭据，不kill、不删、不改。

## GH-ACCOUNT-KEY-001 — 普通账号身份与无覆盖 preflight

- 时间：`2026-08-22 21:20:35 CST`；账号 `chenyiteng`，UID/GID `1003/1003`，组仍为
  `chenyiteng,sudo,labdata`。
- command file：
  `local_scripts/remote_commands/shenzhen_github_account_key_preflight_20260822.sh`；SHA-256
  `75dc1a44b2632670a3543ab5f983ae486f7f79c6cdbc665a6c9f2e6df2221f03`；exit `0`。
- `~/.ssh` 仍为 `700 chenyiteng:chenyiteng`；两把既有 repo deploy key和专用known-hosts均保留。
- 新目标
  `/home/chenyiteng/.ssh/github_yutenji_account_ed25519{,.pub}` 均不存在；preflight marker=`PASS`。

## GH-ACCOUNT-KEY-002 — 生成账号级 Ed25519 keypair

- 时间：`2026-08-22 21:20:55 CST`；command file：
  `local_scripts/remote_commands/shenzhen_github_account_key_generate_20260822.sh`；SHA-256
  `f5bdba1e6885178f9c3a43220b5ff5e566016dd9083e6d55c393363567be3ffd`；exit `0`。
- 精确目标：`/home/chenyiteng/.ssh/github_yutenji_account_ed25519`；private/public mode=`600/644`，
  owner均为`chenyiteng:chenyiteng`。脚本在目标已存在时拒绝执行，因此没有覆盖现有key。
- fingerprint：`SHA256:gMSUm/WoenHcy4U1nBbunKPYpv4mCHgsgHfKR1TmK/w`。
- 只需交给用户的公钥为：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJFIpL20rIIvqgRhc7wO85+4Zm3HuHQKQyBLct3USFAa SZ-H100 GitHub account 2026-08-22
```

- private key未输出、未下载、未进入文档。此key在用户加入 GitHub
  `Settings -> SSH and GPG keys -> New SSH key` 后，可按该GitHub账号已有仓库权限用于Git transport；
  它仍不是GitHub API/OAuth凭据，不能仅凭SSH key调用“创建fork”等账号API。

## ADMIN-AUDIT-001 — 全机容量、用户、GPU与近期会话

- 主要现场：`2026-08-22 21:21:21 CST`；账号`toom`，UID1000，`sudo/adm/lxd/labdata`；sudo只读
  验证通过。command file：
  `local_scripts/remote_commands/shenzhen_admin_readonly_audit_20260822.sh`；SHA-256
  `ebd3e3b51c87fa191cff15ef0dbeb53028feb3e486de349ed2c0d22b352c61ca`；exit `0`。
- 文件系统没有容量或inode压力：

| mount | size / used / available | use | inode use |
|---|---:|---:|---:|
| `/` | `296 GiB / 48 GiB / 235 GiB` | `17%` | `2%` |
| `/home` | `2.3 TiB / 94 GiB / 2.2 TiB` | `5%` | `1%` |
| `/data` | `3.5 TiB / 252 GiB / 3.1 TiB` | `8%` | `1%` |
| `/var/lib/docker` | `344 GiB / 256 KiB / 344 GiB` | `1%` | `1%` |
| `/var/lib/containerd` | `147 GiB / 529 MiB / 147 GiB` | `1%` | `1%` |

- `/` 同文件系统一级实际字节量共约`45.17 GB`；主要是`/usr 18.67 GB`、`/root 9.37 GB`、
  `/tmp 5.62 GB`、`/var 4.08 GB`、`/opt 1.02 GB`。没有“根分区被某用户意外撑爆”的现场证据。
- `/home`一级占用：`chenyiteng 54.84 GB`、`zhangwei 21.24 GB`、`xiongzizhen 13.19 GB`、
  `liwenbo 9.46 GB`、`toom 2.09 GB`。
- `/data`一级占用：`chenyiteng 197.96 GB`、`liwenbo 71.38 GB`、`xiongzizhen 0.47 GB`；
  `zhangwei/toom/shared`当前近空。上述均只读目录元数据，不读取文件正文。
- 21:21的PPO退出过渡点仍看到`chenyiteng`四个EnvWorker合计占全机绝大多数RSS，GPU4–7约
  `68.3–70.1 GiB/卡`；其他用户没有GPU compute process。其余用户即时RSS均很小：`zhangwei`
  约`212 MiB`、`liwenbo`约`42 MiB`，主要只见既有tmux/Codex会话，无异常资源占用。
- `last/who`显示近几天有`chenyiteng、zhangwei、xiongzizhen、liwenbo、toom`正常SSH/tmux会话；
  本检查不读取命令历史，因此只据登录、进程和资源元数据判断，不能声称还原每个人执行过的每条命令。

## ADMIN-AUDIT-002 — PPO退出后资源与系统异常计数

- `2026-08-22 21:22:54–21:25:25 CST`复核；PPO由主线按用户授权停止，不是本审计发出的signal。
- command files：
  - `shenzhen_admin_alert_summary_20260822.sh`，SHA-256
    `945e889d8a1cc924963668b12660886337b947853bae8b891de7d6b348df807a`；
  - `shenzhen_admin_kernel_counter_20260822.sh`，SHA-256
    `203855c52becf97ed6903ab1ce76c72ed30b7d3028bf0c3e8d645e55338db0d1`。
- PPO/Ray释放后`MemAvailable=2,091,777,020 KiB≈1.948 TiB`，swap仅约22MiB used；当时全机GPU
  compute process为空，说明此前约1.8TiB内存压力来自本轮PPO而不是其他用户常驻负载。
- failed systemd units=`0`。
- 当前boot的`dmesg`与近3天journal中：OOM / killed-process / no-space / disk-full /
  read-only-filesystem / filesystem-I/O / NVMe-error匹配均为`0`。
- 当前boot有6条NVIDIA Xid行，全部集中在`2026-08-21 11:37:31`和`11:44:04 UTC`的physical GPU0：
  Xid13 `Illegal Instruction Parameter`与随后的Xid43（PID `1235975/1237399`）。这与ACT阶段已经定位的
  CuRobo fused LBFGS H100 illegal-instruction事件时间和GPU一致；之后没有新Xid。它不是磁盘或内存事故。

## PPO-TERMINAL-READONLY — Step47与进程释放复核

- `2026-08-22 21:28:16 CST`只读command：
  `local_scripts/remote_commands/shenzhen_ppo_formal100_terminal_summary_20260822.sh`；SHA-256
  `9dd69748a9023b455b8f7c2d5bce39ffc1914fc7beef87d8ebd1558f45c11051`；exit `0`。
- 最新完整`Global Step 47/100`：train success/return=`0.89453125`，KL=`0.014`，clip fraction=`0.060`，
  grad norm=`32.559`，critic explained variance=`0.442`，step/rollout/actor=`1558.0/1532.5/23.156s`；
  全部finite。
- driver终态为Step48首个rollout epoch完成后收到主线既定升级的`SIGTERM`；没有新训练异常堆栈。
  原PPO driver与已知Ray worker PID survivor=`0`；检查时出现的Ray core与GPU2进程属于随后新启动的π0
  推理，不是PPO残留。
- PPO保留`global_step_10/20/30/40`四个checkpoint；Step47没有新checkpoint，既有产物未删改。

## GH-ACCOUNT-KEY-003 — 网页登记后的显式账号身份验证

- `2026-08-22 22:30:02 CST`，用
  `local_scripts/remote_commands/shenzhen_github_account_key_and_fastwam_preflight_20260822.sh`
  （2,923 bytes；SHA-256
  `85bd4c817f8d46beb9ff22928febbdc2f143c9f16831b1e4aceb972982725817`）执行；exit `0`。
- 服务器公钥指纹仍为
  `SHA256:gMSUm/WoenHcy4U1nBbunKPYpv4mCHgsgHfKR1TmK/w`，与用户网页截图一致；private/public mode仍为
  `600/644`、owner=`chenyiteng:chenyiteng`。
- 未改全局SSH配置，使用显式
  `-i github_yutenji_account_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes`及既有固定GitHub
  known-hosts。`ssh -T`精确返回身份`Yutenji-Nyamu`；账号key访问
  `Yutenji-Nyamu/rlinf_fastwam`的`ls-remote`成功。
- `Yutenji-Nyamu/FastWAM`的`ls-remote`返回`Repository not found`。服务器`gh 2.4.0`已安装，
  但`gh auth status`为未登录；Fast-WAM worktree仍为
  `c63dc9b5384d6637a93cc862dbe2815d0332801d`，tracked diff为空，只保留三个预期RoboTwin symlink，
  remote仍只有official HTTPS `origin`，没有仓库级`core.sshCommand`。

## GH-ACCOUNT-KEY-004 — GitHub CLI设备授权未完成的明确停点

- 两次`gh auth login --hostname github.com --web`均只产生有时限的GitHub device flow；第二次通过
  `local_scripts/remote_commands/shenzhen_github_gh_web_login_device_20260822.sh`
  （290 bytes；SHA-256
  `0b0923b6094248e5698e136756ab12eb54b30da7d6724ba8e5c6dd88209e1ac5`）有界等待600秒，最终exit `124`。
  一次性设备码不作为长期凭据写入账本。
- 结束后`gh auth status`仍为未登录。因此本批次**没有**创建fork、没有添加/修改Fast-WAM remote或
  SSH config、没有push，也没有生成或另行粘贴token。
- 本停点当时，授权完成后的执行脚本已准备但尚未运行（后续成功闭环见`GH-ACCOUNT-KEY-005`）：
  `local_scripts/remote_commands/shenzhen_fastwam_account_fork_remote_push_20260822.sh`
  （当时草案3,763 bytes；SHA-256
  `28375ec99196edf26b486d6eaf6aac9cfb8d1a490356b59ea7a0b470cc8ddda0`）。它会先复核exact local head与
  三个预期symlink，再创建/确认fork、配置**仅此worktree**的账号key传输，并只做普通非force push；
  若远端同名分支已存在但SHA不同则拒绝。

## GH-ACCOUNT-KEY-005 — 账号API授权、Fast-WAM fork与普通push闭环

- `2026-08-22 22:48–22:57 CST`，首次新device flow直连GitHub设备端点时返回
  `unexpected EOF`；依据深圳Paramiko非登录shell不会继承系统网络设置的已知现场，只做一次窄重试：
  source现有`/etc/profile.d/mihomo-proxy.sh`，没有安装或修改网络配置。使用脚本
  `local_scripts/remote_commands/shenzhen_github_gh_web_login_device_proxy_retry1_20260822.sh`
  （354 bytes；SHA-256
  `d2bd45b39a5bcb0fcba5b30368392ea5dee8a1e896434b922dd1af0a10259353`）；用户在GitHub设备页授权后，
  exit `0`并返回`Logged in as Yutenji-Nyamu`。一次性device code不写入长期账本。
- `2026-08-22 22:57:01 CST`执行
  `local_scripts/remote_commands/shenzhen_fastwam_account_fork_remote_push_20260822.sh`
  （3,801 bytes；SHA-256
  `9031206fd8514cd833cb19d38ffd882d087058354e67e7670c55d672c9550c30`）；exit `0`。
- `gh auth status`确认账号`Yutenji-Nyamu`，账号API凭据保存在服务器自己的
  `/home/chenyiteng/.config/gh/hosts.yml`；命令输出仅显示遮蔽token，没有将token写入脚本或账本。
- 通过官方源`yuantianyuan01/FastWAM`创建fork `Yutenji-Nyamu/FastWAM`；第一次SSH探针即看到
  fork `main@7faa71108368fbb3b6885649f112af607427a2d4`。
- push前守卫确认local worktree为
  `codex/sz-fastwam-dvac-observe@c63dc9b5384d6637a93cc862dbe2815d0332801d`，tracked/index diff为空，
  只有三个预期RoboTwin symlink。随后添加repo-local `personal` remote：
  `git@github.com:Yutenji-Nyamu/FastWAM.git`，并只在此worktree的`core.sshCommand`绑定账号key、
  `IdentitiesOnly`和固定known-hosts；既有RLinf deploy-key remotes及全局SSH配置未改。
- 普通非force push新建远端分支`personal/codex/sz-fastwam-dvac-observe`；最终remote head精确为
  `c63dc9b5384d6637a93cc862dbe2815d0332801d`，本地分支已跟踪该远端。push后tracked tree仍无diff，
  三个预期symlink保持未跟踪，没有删除或覆盖文件。
