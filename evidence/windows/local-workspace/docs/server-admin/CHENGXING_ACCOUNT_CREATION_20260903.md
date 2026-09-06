# chengxing 账号创建记录（2026-09-03）

- 授权：用户明确要求创建 `chengxing`，设置其指定口令并授予 sudo。
- 连接：使用固定 host-key 的 Paramiko 管理员路线，以 `toom@admin` 完成身份探针；口令仅在进程内提供，未写入文件或 Git。
- 创建：新建 `/home/chengxing`，shell 为 `/bin/bash`；加入 `sudo` 与服务器通用 `labdata` 组。
- 存储：`/home/chengxing` 与 `/data/chengxing` 均为 `0700 chengxing:chengxing`；`~/data -> /data/chengxing`，`~/shared -> /data/shared`。
- 验证：新账号密码 SSH 实登成功；`id` 为 `uid=1010(chengxing) gid=1011(chengxing)`，补充组为 `sudo,labdata`；非交互 sudo 验证成功。
- 边界：未修改其他账号、服务、训练进程或共享数据。

