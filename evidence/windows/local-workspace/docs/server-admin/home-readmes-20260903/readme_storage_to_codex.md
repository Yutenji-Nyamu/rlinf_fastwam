# 共享服务器存储说明

> 本文件是 `/home/readme_to_codex.md` 的存储子说明。以下是使用建议，不构成强制规范，具体以人类判断和实时状态为准。

## 当前存储布局

```text
nvme0（系统盘，约 3.5 TiB）
├── EFI                         1 GiB
├── /boot                       2 GiB
└── ubuntu-vg（LVM 存储池）
    ├── /                       300 GiB
    ├── /home                  2355 GiB（约 2.3 TiB）
    ├── /var/lib/docker         350 GiB
    ├── /var/lib/containerd     150 GiB
    └── 未分配 VG 空间          约 419 GiB

nvme1（数据盘，约 3.5 TiB）
└── /data                       持久个人数据和共享数据
```

`nvme0` 上的 LVM 让各用途保持文件系统隔离。保留的约 419 GiB VG 空间不是可直接进入的目录，可供管理员以后扩展现有逻辑卷或新建逻辑卷。原 `/scratch` 逻辑卷已于 2026-08-21 删除，空间主要用于扩展 `/home`。

## 用户入口

```text
~            /home/<用户名>       项目、个人环境、依赖、配置和默认缓存
~/data    -> /data/<用户名>       个人长期数据、大型数据集、模型和结果
~/shared  -> /data/shared         团队共享
```

用户无需自己挂载磁盘。`~/data` 和 `~/shared` 是软链接，用起来与普通目录相同，但实际占用独立 `nvme1` 的 `/data` 空间。系统不再提供 `~/scratch`。

## 放置建议

- GitHub 项目、小型或中型项目、个人 Python/Conda 环境可直接放在自己的家目录，例如 `~/projects`；也可把包含大数据的完整项目放在 `~/data/<项目名>`。
- 需要长期保留的大型数据集、模型权重、checkpoint 和结果优先放在 `~/data`。
- pip、Hugging Face、PyTorch、npm、Conda 等工具若未单独配置，其默认用户缓存通常位于 `~/.cache`、`~/.npm`、`~/.conda` 等位置，因此占用 `/home` 的 2.3 TiB 空间。
- 如果希望项目与大资源完整放在数据盘，可在 `~/data/<项目名>` 中创建环境，并通过项目配置或环境变量把缓存指向该项目目录；不要假定所有安装程序都会跟随当前工作目录。
- `/home` 与 `/data` 当前都未启用每用户磁盘配额。多人共用时应定期用 `df -h ~ ~/data` 和 `du -sh ~ ~/data` 检查容量。

## 共享区

```text
~/shared/
├── datasets/   公共数据集，普通用户当前只读
├── projects/   labdata 组的项目协作目录
└── transfer/   临时文件交换目录
```

个人家目录和 `~/data` 默认为本人私有；需要团队协作的内容使用 `~/shared`，具体目录组织由团队决定。

## 系统环境与容器存储

- CUDA、NVIDIA 驱动和系统库位于公共系统目录，由用户共用；通过系统包管理器安装的库主要占用根文件系统。
- 个人 Python/Conda 环境由各用户管理，通常占用 `/home`；放在 `~/data` 时则占用独立数据盘。
- Docker 和 containerd 分别使用独立逻辑卷，容器镜像和运行时数据不会占用 `/home`；普通用户目前未加入 `docker` 组。

确认路径实际落在哪个文件系统时，可使用：

```bash
pwd -P
df -h . ~ ~/data
readlink -f ~/data ~/shared
```

## 2026-08-21 调整记录

- 修改前的 GPT、LVM、`fstab`、挂载与权限记录保存在 `/data/toom/admin-backups/home-expansion-20260821-0325`。
- 删除了已确认无用户数据、无占用进程的 `/scratch` 文件系统和 `scratch-lv`。
- `/home` 从 100 GiB 在线扩展为 2355 GiB，现有家目录内容原地保留。
- 删除了五个用户原有的 `~/scratch` 链接；`~/data` 与 `~/shared` 保持原目标不变。
