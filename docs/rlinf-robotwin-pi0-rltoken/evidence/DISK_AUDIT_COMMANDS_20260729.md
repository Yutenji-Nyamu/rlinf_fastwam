# RLT 前置磁盘归属审计：精确命令与只读证据

> 日期：2026-07-29
> 观察窗口：2026-07-29 15:37:13–15:45:53（Asia/Shanghai，UTC+08:00）
> 性质：服务器只读审计；本文记录实际调用的 A–P 命令体、返回状态、失败与窄修复。
> 结论边界：本轮没有删除、移动、改名、覆盖或编辑任何服务器文件。

## 1. 共用连接与凭据边界

所有远端命令都通过本地
`local_scripts/remote_exec_autodl.py` 调用。SSH 密码仅在每次 PowerShell 调用的当前进程中
注入 `SEETA_SSH_PASSWORD`，调用结束后立即移除；本文不记录密码值，也没有把密码写入脚本、
文档、命令参数、服务器或仓库。连接前由 helper 校验固定 host-key，再进行 password
authentication。

除 G 外，下面代码块都是交给 helper 的 PowerShell here-string 源命令体。G 在本地
argument parsing 阶段失败，远端从未执行；因此 G 记录的是实际尝试传入的 here-string
原文，而不是一条已经抵达服务器的命令。I 已抵达服务器，主体只读命令执行完成，但其中
一条 `awk` 管道失败。

## 2. A–P 精确命令时间线

### A. 身份、资源、进程与 Git 基线

- 远端观察时间：`2026-07-29T15:37:13+08:00`
- 返回：helper `rc=0`
- 结果用途：确认服务器身份、磁盘、RAM、GPU、相关进程及两个历史仓库的既有状态。

```bash
set -u
export LC_ALL=C
date --iso-8601=seconds
id
hostname
printf 'ROOT_DF\n'
df -h /root/autodl-tmp
printf 'MEMORY\n'
free -h
printf 'GPU\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'RELEVANT_PROCESSES\n'
ps -eo pid,ppid,lstart,etime,rss,args --sort=-rss | awk 'NR==1 || /ray|robotwin|python.*(main|train|sft|rlt|ppo|grpo|dsrl)/ {print}' | head -80
printf 'RLINF_GIT\n'
git -C /root/autodl-tmp/RLinf rev-parse --show-toplevel HEAD 2>/dev/null || true
git -C /root/autodl-tmp/RLinf status --short --branch 2>/dev/null || true
printf 'ROBOTWIN_GIT\n'
git -C /root/autodl-tmp/RoboTwin rev-parse --show-toplevel HEAD 2>/dev/null || true
git -C /root/autodl-tmp/RoboTwin status --short --branch 2>/dev/null || true
```

### B. 第一版逐目录 `du + stat`

- 远端观察时间：`2026-07-29T15:37:36+08:00`
- 返回：helper `rc=0`
- 结果用途：首次取得 RLinf、RLinf logs、RoboTwin 和 RoboTwin policy 的顶层大小与 mtime。
- 局限：命令成功，但把包含空格的 `stat %y` 与路径共同送入数值排序后，输出不便可靠关联；
  未据此单独下结论，随后用 C 将 bytes/path 与 mtime/path 分开复测。

```bash
set -u
export LC_ALL=C
printf 'AUDIT_TIME\n'
date --iso-8601=seconds
printf 'RLINF_TOP_BYTES\n'
find /root/autodl-tmp/RLinf -mindepth 1 -maxdepth 1 -xdev -print0 | while IFS= read -r -d '' p; do b=$(du -sx --block-size=1 -- "$p" | awk '{print $1}'); m=$(stat -c '%y' -- "$p"); printf '%15s\t%s\t%s\n' "$b" "$m" "$p"; done | sort -nr
printf 'RLINF_LOG_RUN_BYTES\n'
find /root/autodl-tmp/RLinf/logs -mindepth 1 -maxdepth 1 -xdev -print0 | while IFS= read -r -d '' p; do b=$(du -sx --block-size=1 -- "$p" | awk '{print $1}'); m=$(stat -c '%y' -- "$p"); printf '%15s\t%s\t%s\n' "$b" "$m" "$p"; done | sort -nr
printf 'ROBOTWIN_TOP_BYTES\n'
find /root/autodl-tmp/RoboTwin -mindepth 1 -maxdepth 1 -xdev -print0 | while IFS= read -r -d '' p; do b=$(du -sx --block-size=1 -- "$p" | awk '{print $1}'); m=$(stat -c '%y' -- "$p"); printf '%15s\t%s\t%s\n' "$b" "$m" "$p"; done | sort -nr
printf 'ROBOTWIN_POLICY_BYTES\n'
find /root/autodl-tmp/RoboTwin/policy -mindepth 1 -maxdepth 1 -xdev -print0 | while IFS= read -r -d '' p; do b=$(du -sx --block-size=1 -- "$p" | awk '{print $1}'); m=$(stat -c '%y' -- "$p"); printf '%15s\t%s\t%s\n' "$b" "$m" "$p"; done | sort -nr
```

### C. 逐目录 bytes/path 与 mtime/path 分离复测

- 远端观察时间：`2026-07-29T15:38:17+08:00`
- 返回：helper `rc=0`
- 结果用途：形成后续容量归属的可靠顶层清单。

```bash
set -u
export LC_ALL=C
printf 'TIME\n'; date --iso-8601=seconds
printf 'RLINF_LOG_BYTES_PATH\n'
find /root/autodl-tmp/RLinf/logs -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr
printf 'RLINF_LOG_MTIME_PATH\n'
find /root/autodl-tmp/RLinf/logs -mindepth 1 -maxdepth 1 -xdev -printf '%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort
printf 'ROBOTWIN_TOP_BYTES_PATH\n'
find /root/autodl-tmp/RoboTwin -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr
printf 'ROBOTWIN_POLICY_BYTES_PATH\n'
find /root/autodl-tmp/RoboTwin/policy -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr
printf 'ROBOTWIN_TARGET_MTIME\n'
find /root/autodl-tmp/RoboTwin/policy /root/autodl-tmp/RoboTwin/assets -mindepth 1 -maxdepth 1 -xdev -printf '%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort
```

### D. 四个 RLinf 历史 run 的 DCP 与大文件归属

- 远端观察时间：`2026-07-29T15:38:43+08:00`
- 返回：helper `rc=0`
- 结果用途：逐 run 拆出 immediate child、checkpoint/DCP、超过 512 MB 的文件和小型元数据。

```bash
set -u
export LC_ALL=C
printf 'TIME\n'; date --iso-8601=seconds
for run in /root/autodl-tmp/RLinf/logs/2026*; do
  [ -d "$run" ] || continue
  printf 'RUN\t%s\n' "$run"
  printf 'IMMEDIATE_BYTES\n'
  find "$run" -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr
  printf 'CHECKPOINT_DIRS\n'
  find "$run" -xdev -type d \( -name 'global_step_*' -o -iname '*checkpoint*' -o -iname '*dcp*' \) -print0 | while IFS= read -r -d '' d; do du -sx --block-size=1 -- "$d"; stat -c '%y\t%n' -- "$d"; done
  printf 'LARGE_FILES_GT_512M\n'
  find "$run" -xdev -type f -size +512M -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -nr | head -80
  printf 'METADATA_FILES\n'
  find "$run" -xdev -maxdepth 4 -type f \( -iname '*config*' -o -name '*.yaml' -o -name '*.json' -o -name '*.log' -o -name 'events.out.tfevents*' \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -k3 | head -120
 done
```

### E. 历史 run 的原始启动命令、末端指标与 DCP step

- 远端观察时间：`2026-07-29T15:39:13+08:00`
- 返回：helper `rc=0`
- 结果用途：将目录名称映射到 PPO/GRPO smoke/formal，并确认实际完成或中断位置。

```bash
set -u
export LC_ALL=C
printf 'TIME\n'; date --iso-8601=seconds
for run in /root/autodl-tmp/RLinf/logs/2026*; do
  [ -d "$run" ] || continue
  printf 'RUN\t%s\n' "$run"
  printf 'COMMAND\n'; sed -n '1,12p' "$run/command.txt" 2>/dev/null || true
  printf 'METRICS_HEAD\n'; sed -n '1,3p' "$run/metrics.log" 2>/dev/null || true
  printf 'METRICS_TAIL\n'; tail -3 "$run/metrics.log" 2>/dev/null || true
  printf 'LOG_FINAL_SIGNALS\n'; grep -E 'Global Step|global_step|success|Training completed|Finished|Saving checkpoint|Step [0-9]+' "$run/run_embodiment.log" 2>/dev/null | tail -20 || true
  printf 'DCP_STEPS\n'; find "$run" -xdev -type d -name 'global_step_*' -printf '%f\n' | sort -V | tr '\n' ' '; printf '\n'
done
```

### F. RoboTwin 四个大目标的第一轮深查

- 远端观察时间：`2026-07-29T15:39:43+08:00`
- 返回：helper `rc=0`
- 结果用途：分别检查 `Motus_old`、当前 `Motus`、`ACT` 和 `assets` 的 immediate child、
  大文件、symlink 与 metadata。
- 记录说明：命令成功；因输出很长，客户端展示被截断。后续 H–O 用窄命令补齐所需证据。

```bash
set -u
export LC_ALL=C
printf 'TIME\n'; date --iso-8601=seconds
for target in \
 /root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133 \
 /root/autodl-tmp/RoboTwin/policy/Motus \
 /root/autodl-tmp/RoboTwin/policy/ACT \
 /root/autodl-tmp/RoboTwin/assets; do
  printf 'TARGET\t%s\n' "$target"
  stat -c 'TYPE=%F MODE=%A SIZE=%s MTIME=%y PATH=%n' -- "$target"
  printf 'IMMEDIATE_BYTES\n'
  find "$target" -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr | head -80
  printf 'LARGE_FILES_GT_256M\n'
  find "$target" -xdev -type f -size +256M -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -nr | head -100
  printf 'SYMLINKS\n'
  find "$target" -xdev -type l -printf '%p -> %l\n' | head -80
  printf 'METADATA_CANDIDATES\n'
  find "$target" -xdev -maxdepth 4 -type f \( -iname 'README*' -o -iname '*.md' -o -iname '*.yaml' -o -iname '*.yml' -o -iname '*.json' -o -iname '*.sh' -o -iname '*.txt' -o -iname '*.log' \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -k3 | head -160
done
```

### G. 失败 1：本地 argument parsing

- 远端观察时间：无；命令没有抵达服务器。
- 返回：helper CLI `rc=2`
- 错误：`remote_exec_autodl.py: error: unrecognized arguments: bytes=%.0f min=%.0f max=%.0f...`
- 原因：PowerShell native argv 处理破坏了 `awk printf` 中的内嵌双引号，使本应为一个
  remote command argument 的 here-string 被拆成额外参数。
- 影响：零远端执行、零服务器写入。
- 窄修复：仅将 `awk END{printf ...}` 改为 H 中不含内嵌格式字符串的
  `END{print n,s,lo,hi}`；其余目标和读取范围不变。

实际尝试传入的 here-string 原文：

```bash
set -u
export LC_ALL=C
base=/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133
printf 'TIME\n'; date --iso-8601=seconds
printf 'README\n'; sed -n '1,180p' "$base/README.md"
printf 'DEPLOY_CONFIG\n'; sed -n '1,220p' "$base/deploy_policy.yml"
printf 'MAJOR_CHILDREN\n'
for d in "$base"/logs_single_20260602_170538 "$base"/logs_single_20260601_082941 "$base"/logs_his; do
 printf 'DIR\t%s\n' "$d"
 find "$d" -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr | head -100
 printf 'PT_SUMMARY\n'
 find "$d" -xdev -type f -name '*.pt' -printf '%s\n' | awk '{n+=1;s+=$1;if(n==1||$1<lo)lo=$1;if($1>hi)hi=$1} END{printf \"count=%d bytes=%.0f min=%.0f max=%.0f\n\",n,s,lo,hi}'
 printf 'PT_PARENTS\n'
 find "$d" -xdev -type f -name '*.pt' -printf '%h\n' | sort | uniq -c | sort -nr | head -80
done
printf 'TTS_OPD_TEXT_HITS\n'
find "$base" -xdev -type f -size -2M \( -name '*.py' -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' -o -name '*.md' -o -name '*.txt' -o -name '*.log' \) -print0 | xargs -0 grep -IinE 'distill|online policy distill|OPD|TTS|VTTS|teacher|student|winner' 2>/dev/null | head -160
```

### H. G 的窄修复复测

- 远端观察时间：`2026-07-29T15:41:00+08:00`
- 返回：helper `rc=0`
- 结果用途：读取 Motus README/config，拆出两个 OPD run 与 `logs_his`，统计 `.pt` 数量和
  parent，并核对 TTS/OPD 文本证据。

```bash
set -u
export LC_ALL=C
base=/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133
printf 'TIME\n'; date --iso-8601=seconds
printf 'README\n'; sed -n '1,180p' "$base/README.md"
printf 'DEPLOY_CONFIG\n'; sed -n '1,220p' "$base/deploy_policy.yml"
printf 'MAJOR_CHILDREN\n'
for d in "$base"/logs_single_20260602_170538 "$base"/logs_single_20260601_082941 "$base"/logs_his; do
 printf 'DIR\t%s\n' "$d"
 find "$d" -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr | head -100
 printf 'PT_SUMMARY count bytes min max\n'
 find "$d" -xdev -type f -name '*.pt' -printf '%s\n' | awk '{n+=1;s+=$1;if(n==1||$1<lo)lo=$1;if($1>hi)hi=$1} END{print n,s,lo,hi}'
 printf 'PT_PARENTS\n'
 find "$d" -xdev -type f -name '*.pt' -printf '%h\n' | sort | uniq -c | sort -nr | head -80
done
printf 'TTS_OPD_TEXT_HITS\n'
find "$base" -xdev -type f -size -2M \( -name '*.py' -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' -o -name '*.md' -o -name '*.txt' -o -name '*.log' \) -print0 | xargs -0 grep -IinE 'distill|OPD|TTS|VTTS|teacher|student|winner' 2>/dev/null | head -160
```

### I. 失败 2：扩展名聚合中的 `awk` quoting

- 远端观察时间：`2026-07-29T15:41:34+08:00`
- 返回：helper 外层 `rc=0`；其中 `MOTUS_LOGS_HIS_EXT_SUM` 管道失败。
- 错误：`awk: line 1: runaway string constant`
- 原因：`sub(/^.*\./,\"\",e)` 的引号经本地到远端的命令参数链后没有形成有效 awk 字符串。
- 影响：该扩展名聚合没有产生有效结果；同一命令体中的其他只读 `find`、`du`、`sed`
  已执行并输出。命令没有启用 `pipefail`，因此最后一个成功命令使 helper 外层仍为 `rc=0`。
- 窄修复：J 不再动态解析扩展名，而是对固定扩展列表逐项运行
  `find -iname "*.$ext"`，保留相同根目录与只读范围。

```bash
set -u
export LC_ALL=C
old=/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133
cur=/root/autodl-tmp/RoboTwin/policy/Motus
act=/root/autodl-tmp/RoboTwin/policy/ACT
printf 'TIME\n'; date --iso-8601=seconds
printf 'MOTUS_LOGS_HIS_TOP_FILES\n'
find "$old/logs_his" -xdev -type f -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -nr | head -80
printf 'MOTUS_LOGS_HIS_EXT_SUM bytes count ext\n'
find "$old/logs_his" -xdev -type f -printf '%s %f\n' | awk '{e=$2;sub(/^.*\./,\"\",e);s[e]+=$1;c[e]+=1} END{for(e in s)print s[e],c[e],e}' | sort -nr | head -40
printf 'MOTUS_CURRENT_CHILDREN\n'
find "$cur" -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr
printf 'MOTUS_CURRENT_WEIGHT_FILES\n'
find "$cur" -xdev -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.ckpt' \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -nr
printf 'ACT_CHILDREN\n'
find "$act" -mindepth 1 -maxdepth 1 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr
printf 'ACT_PROCESSED_SETS\n'
find "$act/processed_data" -mindepth 1 -maxdepth 3 -type d -exec du -sx --block-size=1 -- {} + | sort -nr | head -80
printf 'ACT_HDF5_SUM count bytes min max\n'
find "$act" -xdev -type f -name '*.hdf5' -printf '%s\n' | awk '{n+=1;s+=$1;if(n==1||$1<lo)lo=$1;if($1>hi)hi=$1} END{print n,s,lo,hi}'
printf 'ACT_MODEL_FILES\n'
find "$act" -xdev -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.ckpt' \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -nr | head -80
printf 'ACT_CONFIGS\n'; sed -n '1,180p' "$act/SIM_TASK_CONFIGS.json"; sed -n '1,160p' "$act/deploy_policy.yml"; sed -n '1,120p' "$act/train.sh"; sed -n '1,120p' "$act/process_data.sh"
```

### J. I 的逐扩展窄修复与依赖关系核验

- 远端观察时间：`2026-07-29T15:43:01+08:00`
- 返回：helper `rc=0`
- 结果用途：固定扩展统计、两个 OPD run 的首末 checkpoint 和 success 信号、ACT checkpoint
  hash/inode、RLT 实际 assets 路径及两份 assets 的 inode/realpath。

```bash
set -u
export LC_ALL=C
old=/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133
act=/root/autodl-tmp/RoboTwin/policy/ACT
printf 'TIME\n'; date --iso-8601=seconds
printf 'MOTUS_LOGS_HIS_EXT_COUNTS_AND_BYTES\n'
for ext in png jpg jpeg mp4 npz csv log json pt pth; do
  vals=$(find "$old/logs_his" -xdev -type f -iname "*.$ext" -printf '%s\n' | awk '{n+=1;s+=$1} END{print n+0,s+0}')
  printf '%s\t%s\n' "$ext" "$vals"
done
printf 'MOTUS_RUN_ENDPOINTS\n'
for d in "$old"/logs_single_20260602_170538 "$old"/logs_single_20260601_082941; do
  printf 'RUN\t%s\n' "$d"
  find "$d" -xdev -type f -name '*.pt' -printf '%T@\t%s\t%p\n' | sort -n | head -2
  find "$d" -xdev -type f -name '*.pt' -printf '%T@\t%s\t%p\n' | sort -n | tail -2
  printf 'SUCCESS_SIGNALS\n'
  grep -Eih 'success rate|success_once|success count|successes|episode.*success|succ=' "$d/turn_switch.log" 2>/dev/null | tail -20 || true
done
printf 'ACT_HASHES\n'
sha256sum "$act"/act_ckpt/act-beat_block_hammer/demo_clean-50/*.ckpt
printf 'ACT_INODES\n'
stat -c '%i\t%h\t%s\t%y\t%n' "$act"/act_ckpt/act-beat_block_hammer/demo_clean-50/*.ckpt
printf 'RLT_ENV_PATH_REFERENCES\n'
grep -RInE '/root/autodl-tmp/RoboTwin(_RLinf)?|RoboTwin_RLinf' /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/embodiment/config /root/autodl-tmp/RLinf_rlt_pi0_robotwin/evaluations 2>/dev/null | head -100
printf 'ASSET_RELATIONS\n'
for p in /root/autodl-tmp/RoboTwin/assets /root/autodl-tmp/RoboTwin_RLinf/assets; do
  if [ -e "$p" ] || [ -L "$p" ]; then stat -c 'TYPE=%F INODE=%i LINKS=%h SIZE=%s MTIME=%y PATH=%n' "$p"; readlink -f "$p"; du -sx --block-size=1 -- "$p"; else printf 'MISSING\t%s\n' "$p"; fi
done
printf 'ASSET_SAMPLE_INODES\n'
for rel in embodiments/aloha-agilex/config.yml objects/001_bottle/model_data0.json; do
  stat -c '%d\t%i\t%h\t%s\t%n' "/root/autodl-tmp/RoboTwin/assets/$rel" "/root/autodl-tmp/RoboTwin_RLinf/assets/$rel" 2>/dev/null || true
done
```

### K. ACT checkpoint 文件清单与 assets manifest

- 远端观察时间：`2026-07-29T15:43:55+08:00`
- 返回：helper `rc=0`
- 结果用途：确认 ACT checkpoint/plot/stats 构成，以及两份 assets 的 file/dir 数量和抽样 hash。

```bash
set -u
export LC_ALL=C
ck=/root/autodl-tmp/RoboTwin/policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50
printf 'TIME\n'; date --iso-8601=seconds
printf 'ACT_CKPT_ALL_FILES\n'
find "$ck" -maxdepth 2 -type f -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -k3
printf 'ACT_SMALL_TEXT_CONTENT\n'
for f in "$ck"/*.txt "$ck"/*.json "$ck"/*.log; do [ -f "$f" ] || continue; printf 'FILE\t%s\n' "$f"; sed -n '1,160p' "$f"; done
printf 'ASSET_MANIFEST_COUNTS\n'
for p in /root/autodl-tmp/RoboTwin/assets /root/autodl-tmp/RoboTwin_RLinf/assets; do
 printf 'PATH\t%s\n' "$p"
 find "$p" -xdev -type f -printf '%P\t%s\n' | sort | sha256sum
 find "$p" -xdev -type f | wc -l
 find "$p" -xdev -type d | wc -l
done
printf 'ASSET_SAMPLE_HASHES\n'
sha256sum /root/autodl-tmp/RoboTwin/assets/embodiments/aloha-agilex/config.yml /root/autodl-tmp/RoboTwin_RLinf/assets/embodiments/aloha-agilex/config.yml /root/autodl-tmp/RoboTwin/assets/objects/001_bottle/model_data0.json /root/autodl-tmp/RoboTwin_RLinf/assets/objects/001_bottle/model_data0.json
```

### L. 两份 assets 的 path+size 差异

- 远端观察时间：`2026-07-29T15:44:15+08:00`
- 返回：helper `rc=0`
- 结果用途：不用创建临时文件，比较两份 assets 的相对路径和 byte size。

```bash
set -u
export LC_ALL=C
printf 'TIME\n'; date --iso-8601=seconds
printf 'ASSET_PATH_SIZE_DIFFS\n'
comm -3 <(cd /root/autodl-tmp/RoboTwin/assets && find . -type f -printf '%P\t%s\n' | sort) <(cd /root/autodl-tmp/RoboTwin_RLinf/assets && find . -type f -printf '%P\t%s\n' | sort) | head -100
printf 'ASSET_DIFF_LINE_COUNT\n'
comm -3 <(cd /root/autodl-tmp/RoboTwin/assets && find . -type f -printf '%P\t%s\n' | sort) <(cd /root/autodl-tmp/RoboTwin_RLinf/assets && find . -type f -printf '%P\t%s\n' | sort) | wc -l
```

### M. `beat_block_hammer` 数据与产物位置

- 远端观察时间：`2026-07-29T15:44:56+08:00`
- 返回：helper `rc=0`
- 结果用途：判断 ACT processed data 是否仍有原始可重建源。

```bash
set -u
export LC_ALL=C
printf 'TIME\n'; date --iso-8601=seconds
printf 'BEAT_BLOCK_HAMMER_PATHS\n'
find /root/autodl-tmp -xdev \( -iname '*beat_block_hammer*' -o -iname '*beat-block-hammer*' \) -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null | head -200
printf 'ROBOTWIN_DATA_CHILDREN\n'
find /root/autodl-tmp/RoboTwin/data -mindepth 1 -maxdepth 4 -xdev -exec du -sx --block-size=1 -- {} + | sort -nr | head -100
```

### N. ACT 原始 50 条数据结构

- 远端观察时间：`2026-07-29T15:45:12+08:00`
- 返回：helper `rc=0`
- 结果用途：确认 raw `demo_clean` 下的 pkl/HDF5/instruction/video 仍存在并记录总大小。
- 说明：目标目录没有 `.zip`，因此最后的 ZIP loop 正常无输出，不是失败。

```bash
set -u
export LC_ALL=C
p=/root/autodl-tmp/RoboTwin/data/beat_block_hammer
printf 'TIME\n'; date --iso-8601=seconds
printf 'RAW_CHILDREN\n'
find "$p" -mindepth 1 -maxdepth 3 -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | head -200
printf 'RAW_SIZE\n'; du -sx --block-size=1 -- "$p"
printf 'RAW_ARCHIVE_TEST_IF_ZIP\n'
for z in "$p"/*.zip; do [ -f "$z" ] || continue; unzip -tqq "$z" && printf 'ZIP_OK\t%s\n' "$z"; done
```

### O. 清理候选的 Git 所有权与 ignore 规则

- 远端观察时间：`2026-07-29T15:45:33+08:00`
- 返回：helper `rc=0`
- 结果用途：区分 tracked code、现有用户修改、untracked 目录和 ignored 生成物。

```bash
set -u
export LC_ALL=C
repo=/root/autodl-tmp/RoboTwin
printf 'TIME\n'; date --iso-8601=seconds
printf 'TARGET_GIT_STATUS\n'; git -C "$repo" status --short -- policy/ACT policy/Motus policy/Motus_old_20260618_111133 assets data/beat_block_hammer
printf 'TRACKED_COUNTS\n'
for p in policy/ACT policy/Motus policy/Motus_old_20260618_111133 assets data/beat_block_hammer; do printf '%s\t' "$p"; git -C "$repo" ls-files -- "$p" | wc -l; done
printf 'IGNORE_RULES\n'
for p in policy/ACT/processed_data policy/ACT/act_ckpt policy/Motus_old_20260618_111133 assets data/beat_block_hammer; do git -C "$repo" check-ignore -v "$p" || printf 'NOT_IGNORED\t%s\n' "$p"; done
```

### P. 终态资源与进程刷新

- 远端观察时间：`2026-07-29T15:45:53+08:00`
- 返回：helper `rc=0`
- 结果用途：确认只读审计结束时没有遗留相关进程或 GPU 占用，磁盘状态未因本轮发生变化。

```bash
set -u
export LC_ALL=C
date --iso-8601=seconds
df -h /root/autodl-tmp
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,ppid,etime,rss,args | awk 'NR==1 || /ray|robotwin|python.*(main|train|sft|rlt|ppo|grpo|dsrl)/ {print}' | head -80
```

## 3. 只读结果摘要

### 3.1 现场与执行安全

- A 观察到 `/root/autodl-tmp` 为约 `1.9T`，已用约 `1.2T`、可用 `694G`、使用率
  `63%`；两张 A800 均为 `0 MiB / 0%`。
- A 与 P 都没有发现实际 Ray、RoboTwin 或训练进程；进程筛选只匹配到本次只读 shell/awk。
- `/root/autodl-tmp/RLinf` 和 `/root/autodl-tmp/RoboTwin` 都有既有
  dirty/untracked 状态，因此任何未来清理都必须使用已审计的精确路径，不能整仓清理。
- 本轮没有调用 `rm`、`mv`、`cp`、`rsync`、解压写入、Git 写操作或任何训练命令。

### 3.2 `/root/autodl-tmp/RLinf/logs`

总量为 `145,558,147,072 B`，即 `135.562 GiB`，几乎全部来自四个 2026-07-14 至
2026-07-17 的 π0 PPO/GRPO DCP：

| run | 现场大小 | DCP |
|---|---:|---|
| PPO smoke | `10,402,525,184 B` | step 1：`10,402,308,096 B` |
| PPO baseline | `20,809,404,416 B` | step 10/20，各约 `10.402 GB`；日志到 step 29 |
| GRPO smoke | `10,394,161,152 B` | step 1：`10,393,944,064 B` |
| GRPO formal | `103,952,044,032 B` | step 10–100 共 10 个，每个 `10,393,944,064 B` |

只读审计形成的保守候选是：保留 GRPO step 100、PPO step 20 和所有小型
command/resolved-config/metrics/TensorBoard/resource logs；其余 9 个 GRPO 中间 DCP、
PPO step 10 以及两个 smoke DCP 合计约 `116.177 GiB`。这只是候选，不是删除授权。

### 3.3 `Motus_old`

`/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133` 为
`84,636,626,944 B`，即 `78.824 GiB`：

- `logs_single_20260602_170538`：`48.371 GiB`，40 个 action-expert OPD checkpoint，
  从 `succ0002_ep0004` 到 `succ0080_ep0100`；日志终点 `80/100=80.0%`。
- `logs_single_20260601_082941`：`13.302 GiB`，11 个 checkpoint，终点
  `succ0022_ep0030`；日志终点 `22/30=73.3%`。
- `logs_his`：`17.082 GiB`，主要为 TTS/VTTS 评估图；逐扩展复测得到 25,222 张 PNG，
  另有 50 个 CSV 和 83 个 log，没有 `.pt`。

这证明它混合了 TTS/VTTS 候选选择历史和 OPD/GKD 在线蒸馏 checkpoint，不能把两者笼统
称为同一种产物。若未来每个 OPD run 只保留一个末端 checkpoint，49 个中间 checkpoint
约为 `59.253 GiB`；只清 `logs_his` 的 PNG、保留 CSV/log，约还有 `16.9 GB` 候选。
当前未删除。

### 3.4 ACT

`/root/autodl-tmp/RoboTwin/policy/ACT` 为 `17,710,182,400 B`，即
`16.494 GiB`，属于 2026-05-28 的 `beat_block_hammer clean-50` ACT 模仿学习实验，
不是 Motus/TTS：

- `processed_data`：`15,694,221,312 B`，50 个 HDF5；
- `act_ckpt`：`2,015,543,296 B`，含 best、last 和 epoch 2000/4000/5768/6000；
- 六个 checkpoint 的 SHA256 与 inode 均不同，不是硬链接或字节重复；
- 原始 50 条数据仍在
  `/root/autodl-tmp/RoboTwin/data/beat_block_hammer/demo_clean`，总计
  `320,155,648 B`，因此 processed data 可由现有脚本和原始数据重新生成。

未来若不再运行该 ACT 实验，可优先讨论精确的 generated `processed_data`，以及在保留
best/last/stats/plots 后的中间 epoch checkpoint；`policy/ACT` 有 tracked code 且
`SIM_TASK_CONFIGS.json` 存在用户修改，不能删除整个目录。当前未删除。

### 3.5 两份 assets

- `/root/autodl-tmp/RoboTwin/assets`：`16,665,227,264 B`；
- `/root/autodl-tmp/RoboTwin_RLinf/assets`：`16,665,133,056 B`。

两者是不同 inode 的实体目录，不是 symlink；均有 20,854 个文件和 765 个目录。
path+size manifest 只有 16 行成对差异，即 8 个相对路径，集中在两个 Hugging Face metadata
和六个 CuRobo YAML；抽样的 Aloha config 与 bottle JSON hash 相同。RLT Stage 2 config
默认明确指向 `/root/autodl-tmp/RoboTwin_RLinf`，所以旧
`/root/autodl-tmp/RoboTwin/assets` 不是当前 RLT 的运行依赖；但旧 Motus/ACT/RoboTwin
standalone 仍可能使用它。只有明确退役旧工作流后，旧副本才是约 `15.521 GiB` 的候选。
当前未删除。

## 4. 最终声明

本文是只读归属审计和未来清理讨论的证据，不是清理指令或授权。A–P 期间及本文整理时：

- 未删除任何文件或目录；
- 未移动、改名或覆盖任何产物；
- 未停止任何进程；
- 未启动 smoke、训练或 simulator；
- 未安装依赖；
- 未改变服务器 Git 状态。
