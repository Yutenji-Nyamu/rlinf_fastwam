# 深圳服务器账号与 `/home` README 变更（2026-08-24）

## 已完成

1. `/home/readme_to_codex.md` 的“协作”节现使用用户指定的两条多人协作提示：先联系相应人员或加入微信群；`chenyiteng` 约10月1日前赶 ICLR 27，常用4张卡（例如4567），不得杀死其任务或使剩余卡过少，必要时先联系他。
2. 下载当前 `/home` 根层三份协作文档：`readme_to_codex.md`、`readme_network_to_codex.md`、`readme_storage_to_codex.md`。这里不包含各项目子目录中的第三方 README。
3. 创建 `zhuanghuiping`：`/bin/bash`，组为 `zhuanghuiping sudo labdata`，sudo 合同为 `(ALL : ALL) ALL`；建立私有 `/home/zhuanghuiping`、`/data/zhuanghuiping`，以及 `~/data`、`~/shared` 链接。
4. `zhuanghuiping` 的口令已按用户要求更新，并使用更新后的口令完成固定 host-key SSH 身份探针。口令只在聊天中交付，不写入文档或脚本。
5. 创建普通用户 `guorenjie`：shell `/bin/bash`，仅加入自身组与 `labdata`，明确不在 `sudo` 组；建立私有 `/home/guorenjie`、`/data/guorenjie` 及 `~/data`、`~/shared` 链接，并以该账号完成固定 host-key SSH 登录和 `sudo=denied` 探针。口令只在聊天中交付。
6. 创建普通用户 `qiufuwen`：shell `/bin/bash`，仅加入自身组与 `labdata`，明确不在 `sudo` 组；建立私有 `/home/qiufuwen`、`/data/qiufuwen` 及 `~/data`、`~/shared` 链接，并以该账号完成固定 host-key SSH 登录和 `sudo=denied` 探针。口令只在聊天中交付。
7. 2026-08-25按用户明确授权为 `guorenjie` 增加短路径别名：`/home/guorj -> /home/guorenjie`。执行前确认源目录存在且属主为 `guorenjie:guorenjie`、目标路径不存在；链接由root创建，`readlink -e`解析为原home。它不复制数据、不改变原home的`700`权限。用户重新提供的 `toom` 密码仍被服务器拒绝，因此按用户备选授权通过已验证的 `chenyiteng` sudo只执行这一条链接创建。
8. 2026-08-25创建普通用户 `yanchuhan`：shell `/bin/bash`，仅加入自身组与 `labdata`，明确不在 `sudo` 组；建立私有 `/home/yanchuhan`、`/data/yanchuhan` 及 `~/data`、`~/shared` 链接。随后使用用户指定口令完成固定host-key SSH实登；口令只在聊天交付，不写入文档或脚本。
9. 2026-08-31创建普通用户 `tianfengrui`：shell `/bin/bash`，仅加入自身组与 `labdata`，明确不在 `sudo` 组；建立私有 `/home/tianfengrui`、`/data/tianfengrui` 及 `~/data`、`~/shared` 链接。随后使用用户指定口令完成固定host-key SSH实登，确认`sudo=denied`；口令只在聊天交付，不写入文档或脚本。

本地副本目录和可携带 ZIP 均已随服务器 README 更新：`docs/server-admin/home-readmes-20260824/`、`exports/shenzhen_home_root_readmes_20260824.zip`。

管理员账号 `toom` 的历史口令在本轮认证失败，因此未重试或猜测；上述管理员操作使用已经验证的 `chenyiteng sudo` 路线完成。

## 可复核脚本

- `local_scripts/remote_commands/shenzhen_home_readme_inventory_admin_20260824.sh`
- `local_scripts/remote_commands/shenzhen_create_zhuanghuiping_and_update_main_readme_20260824.sh`
- `local_scripts/remote_commands/shenzhen_zhuanghuiping_identity_probe_20260824.sh`
- `local_scripts/remote_commands/shenzhen_update_zhuanghuiping_password_and_readme_20260824.sh`
- `local_scripts/verified_password_ssh_sudo_chpasswd.py`
- `local_scripts/remote_commands/shenzhen_create_guorenjie_standard_user_20260824.sh`
- `local_scripts/remote_commands/shenzhen_guorenjie_identity_and_no_sudo_probe_20260824.sh`
- `local_scripts/remote_commands/shenzhen_create_qiufuwen_standard_user_20260824.sh`
- `local_scripts/remote_commands/shenzhen_qiufuwen_identity_and_no_sudo_probe_20260824.sh`
- `local_scripts/remote_commands/shenzhen_create_yanchuhan_standard_user_20260825.sh`
- `local_scripts/remote_commands/shenzhen_create_tianfengrui_standard_user_20260831.sh`
- `local_scripts/remote_commands/shenzhen_tianfengrui_identity_probe_20260831.sh`
- `local_scripts/run_sz_verified_user_command.py`

`guorenjie` 创建命令本体已完成后，首轮脚本的只读 `passwd -S` 后检因未带 `sudo` 返回1；未回滚或重复创建。该行已修正，随后由新账号直接登录探针确认身份、目录、共享组和无 sudo 合同全部正确。

`qiufuwen` 创建命令本体完成后，首轮脚本在以管理员普通身份解析其 `700` home 内链接时返回1；链接与账号并未失败。后检改为管理员只读解析，且随后由 `qiufuwen` 直接登录确认身份、目录、共享组和无 sudo 合同全部正确。
