# AutoDL 中国大陆网络：Git/GitHub 短流程

> 目的：把“链路不可达”“认证失败”和“仓库问题”分开；避免一次 Git 操作在大陆线路上
> 无界等待。本文不保存账号、密码、token 或私钥。

## 1. 已验证事实

| 时间 | 现场 | 结论 |
|---|---|---|
| 2026-07-29 | `github.com:443` connect timeout；API HTTP200；push失败 | 主站路由瞬时不可达，不是 commit 或认证结论 |
| 2026-07-30 00:14 | main/API/raw均HTTP200，`ls-remote`成功 | 默认直连恢复 |
| 同次 | ahead5、clean；一次有界 push 用时3秒，upstream变为0/0 | 仓库、remote与已有认证可用 |

当前成功配置是：无 proxy 环境变量、Git `http.version` 默认值、现有 HTTPS remote。
因此不因为一次超时就安装代理、强制 HTTP/1.1、改 remote 或重建 clone。

## 2. 每次同步前的最短诊断

先在服务器做只读检查：

```bash
env | grep -iE '^(http|https|all)_proxy=' || true
git config --get http.version || printf 'DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'api code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://api.github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'raw code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://raw.githubusercontent.com
timeout 15 git ls-remote --heads personal codex/rlt-pi0-robotwin
```

解释：

- main HTTP000/timeout、API200：Git smart-HTTP 所需主站仍不可用；延期，不连续盲试；
- main200但 `ls-remote` 报 permission/authentication：才进入 remote/credential 排查；
- main与 `ls-remote` 都成功：核对 branch、clean、ahead/behind 后只做一次有界同步。

## 3. 有界 push

```bash
test -z "$(git status --short)"
git rev-list --left-right --count '@{upstream}...HEAD'
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git push personal HEAD:codex/rlt-pi0-robotwin
```

push 后必须复核 remote head 与 left/right `0/0`。若60秒内失败，记录 HTTP 探针、
错误文本、HEAD 与 ahead/behind，然后延期；不要在同一坏链路上循环数分钟。

## 4. AutoDL SSH

本工作区沿用已验证 host-key 的 Paramiko password-auth helper：
密码只注入当前进程的 `SEETA_SSH_PASSWORD`，连接结束后在 `finally` 清除；
`look_for_keys=False`、`allow_agent=False`，不把密码写入脚本或日志。OpenSSH
`BatchMode=yes` 不能回答密码提示，因此其失败不能证明 password auth 不可用。

每次网络事件在实施账本记录：时间、main/API/raw结果、`ls-remote`、Git HEAD/dirty/
ahead-behind、实际耗时和是否留下后台进程。
