# 深圳服务器：两块数据盘按用户目录占用

2026-09-05 19:24:17—19:24:41 CST只读现场。用户明确授权整体存储、每位用户、两盘分开查看。固定host-key密码认证管理员toom，身份probe通过，sudo执行低CPU/I/O优先级元数据统计；未读取他人文件内容、未删除/移动/改权限。

命令为`nice -n 19 ionice -c 3 du -x -B1 --max-depth=1 -- /home`及同样的`/data`。两个扫描均rc0，无元数据错误。按顶层用户目录统计实际分配块，不按软链接目标重复计数，不跨文件系统；这不是逐inode UID配额审计。用户目录里代存他人文件仍计入目录总量。动态训练可能在扫描期间新增文件。

## 1. 两盘总览

单位均GiB（1024³字节），不是十进制GB。

| 挂载点 | 总容量 | 已用 | 普通用户可用 | df使用率 | inode使用率 |
|---|---:|---:|---:|---:|---:|
| /home | 2317.48 | 约1070.29 | 1246.15 | 47% | 2% |
| /data | 3519.75 | 2918.90 | 421.99 | 88% | 1% |

`/home`为`/dev/mapper/ubuntu--vg-home--lv`，`/data`为`/dev/nvme1n1`，均ext4。/data比/home紧张；inode余量充足，问题不是文件数量配额已满。19:27 statvfs确认/data还有178.86GiB属于空闲但普通用户不可用的文件系统保留空间（/home为1.04GiB）；它不是某位用户目录占用，本轮未改保留比例。总量−已用不能直接当普通用户可用。精确值见[SZ_STORAGE_SPACE_ACCOUNTING_20260905.json](SZ_STORAGE_SPACE_ACCOUNTING_20260905.json)。

## 2. 每位用户：两个盘分别计数

| 用户目录 | /home GiB | /data GiB |
|---|---:|---:|
| chenyiteng | 53.88 | 1903.46 |
| liwenbo | 149.29 | 944.93 |
| guorenjie | 599.06 | <0.01 |
| zhangwei | 240.89 | <0.01 |
| xiongzizhen | 13.27 | 70.50 |
| chengxing | 8.33 | <0.01 |
| yanchuhan | 3.26 | <0.01 |
| toom | 1.95 | <0.01 |
| qiufuwen | 0.36 | <0.01 |
| zhuanghuiping | <0.01 | <0.01 |
| tianfengrui | <0.01 | <0.01 |

`<0.01`表示小于0.01GiB，不是不存在或完全零字节。/data/shared、lost+found、conda_envs均不足0.01GiB，单列为非用户目录，不归给root训练。/home顶层小说明文件等只有很小残差。

- `/data`：chenyiteng约65.2%、liwenbo约32.4%、xiongzizhen约2.4%的用户目录实际占用；我们的目录是主要占用，不能把磁盘紧张归因于其他用户。
- `/home`：主要为guorenjie599GiB、zhangwei241GiB、liwenbo149GiB；chenyiteng约54GiB。
- 本轮只回答分布，尚未对每个人内部模型/数据/缓存逐层分类，不据目录大小判定某人的文件可删除。

## 3. 与当前训练的关系

19:17预算核对时/data约422GiB，需同时考虑π0.5后续九代checkpoint约241.6GiB、DVAC剩余一代约17.3GiB、GPU6新8×4 smoke两代约34.5GiB。能做本轮smoke，但不能再承诺新BC正式100轮十代约172.8GiB全部写入该盘。

已提出新BC正式写本人/home的方案，用户明确选择**先完成smoke，暂不放正式**；因此未写/home新run、未迁移/删除旧产物、未改变保存频率。后续存储决策须另行明确。
