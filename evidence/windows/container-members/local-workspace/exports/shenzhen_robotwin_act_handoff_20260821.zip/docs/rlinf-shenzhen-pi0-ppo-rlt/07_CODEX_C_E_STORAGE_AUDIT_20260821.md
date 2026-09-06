# Windows Codex：C:/E: 存储现场与迁移边界（2026-08-21）

> 本次只读盘点，没有移动、删除、改名或清理任何目录。逻辑文件大小可能因 WindowsApps 压缩、
> hardlink 等与实际可回收空间不同。

## 1. 结论

**Codex 的主状态已经写到 E:，但迁移没有完全闭合；C: 仍有当前运行依赖、桌面应用、AppData、
Temp 和本项目工作区。现在不能直接删除 `C:\Users\86136\.codex`。**

当前磁盘：

| 卷 | 总量 | 空闲 | 空闲比例 |
|---|---:|---:|---:|
| C: | 200.0 GiB | 37.990 GiB | 18.99% |
| E: | 953.853 GiB | 343.224 GiB | 35.98% |

这次 ACT 本地视频只有 172,523 bytes，本轮文档核心合计约 0.3 MiB；最终 ZIP 也是亚 MiB 量级，
不是 C: 空间风险。真正的大头是历史 Codex state/session、桌面程序/AppData 与已有工作区。

## 2. 官方可配置的 Codex 主状态

OpenAI 官方文档说明：

- `CODEX_HOME` 管理 config、auth、logs、sessions、skills 和 standalone package metadata；默认
  `~/.codex`；
- `CODEX_SQLITE_HOME` 管理 SQLite-backed state，默认跟随 `CODEX_HOME`；
- `CODEX_INSTALL_DIR` 只影响 standalone installer 放置可见 `codex` command 的位置，standalone
  package cache 仍在 `CODEX_HOME/packages/standalone`。它不等同于迁移 Microsoft Store 桌面应用。

官方依据：[Codex environment variables](https://learn.chatgpt.com/docs/config-file/environment-variables)、
[Codex config reference](https://learn.chatgpt.com/docs/config-file/config-reference)。

本机当前 **process-level** 值为：

```text
CODEX_HOME=E:\Codex\home
CODEX_SQLITE_HOME=E:\Codex\home
```

User/Machine 级这两个变量均未设置。因此当前任务确实写 E:；但单独启动的 shell/CLI 如果没有同样的
process 注入，仍可能回退到 `C:\Users\86136\.codex`。

## 3. 当前占用与写入位置

| 位置 | 逻辑大小 | 当前状态/主要内容 |
|---|---:|---|
| `E:\Codex\home` | 39.421 GiB | 正在写；archived sessions 21.465 GiB、sessions 14.679 GiB、`logs_2.sqlite` 2.048 GiB、plugins 0.435 GiB |
| `C:\Users\86136\.codex` | 6.892 GiB | 普通独立目录；最近后代写入 2026-08-18，近 24h 无写入，但仍有 active plugin 依赖 |
| WindowsApps `OpenAI.Codex_26.818.2441.0` | 1.728 GiB | 当前 ChatGPT/Codex 桌面程序和 CLI 本体从 C: 启动 |
| `AppData\Local\OpenAI\Codex` | 0.626 GiB | bin 0.357 GiB、runtimes 0.269 GiB |
| `AppData\Roaming\Codex` | 0.329 GiB | Chromium profile/cache，当前仍写 |
| `AppData\Local\Codex\Logs` | 0.080 GiB | 76 个桌面日志文件，今日仍写 |
| `%TEMP%` | 0.289 GiB | 其中名称明确为 Codex/OpenAI/ChatGPT 的约 0.065 GiB |
| `C:\Users\86136\Documents\rl` | 1.355 GiB | 当前项目工作区；本任务新增文档、视频、ZIP 默认写这里 |

`C:\Users\86136\.codex` 与 `E:\Codex\home` 都是普通目录，不是 junction/symlink。

## 4. 为什么 C: 的旧 `.codex` 现在不能删

现场进程中，`extension-host.exe` 仍从 `C:\Users\86136\.codex\plugins\...` 运行；同时 C、E 两份
`chrome-native-hosts-v2.json` 都仍把 `browserClientPath`、`codexCliPath`、`codexHome` 与
`extensionHostPath` 指向 C: 的 `.codex`。

所以现状是：

```text
主要 sessions / archives / SQLite state  -> E:\Codex\home
部分 plugin / native-host 运行路径         -> C:\Users\86136\.codex
桌面应用 / AppData / Temp / C工作区         -> C:
```

直接删除或改名 C `.codex` 可能让 extension host、Chrome native host 或独立 CLI 失效；仅设置
`CODEX_HOME` 也不会自动迁移 WindowsApps 与 AppData。

## 5. 哪些能放 E:，哪些仍会进 C:

### 可以优先放 E:

- `CODEX_HOME` / `CODEX_SQLITE_HOME` 管辖的 sessions、archives、state/log SQLite、skills 和多数
  package/plugin state；当前已经主要落在 E。
- 新的大型项目工作区、repo clone、资料下载与长期 export；最稳妥的方式是在 E: 新建目录后从那个
  路径打开新 Codex task，而不是让 C: 工作区生成大产物再搬。
- 用户主动选择的模型、数据、视频和大压缩包；本项目的真正大模型/数据仍只放深圳服务器 `/data`。

### 仍可能写 C:

- Microsoft Store/WindowsApps 的桌面应用本体和更新；
- Local/Roaming AppData、桌面日志、Chromium profile/cache；
- Local OpenAI runtime/bin；
- `%TEMP%`、剪贴板/附件临时副本和某些临时 worktree；
- 当前位于 C: 的项目工作区；
- 目前仍写死到 C `.codex` 的 native-host/plugin 路径。

`TEMP/TMP` 可以在 Windows 层改到 E:，但这是影响全部应用的系统性选择，不建议仅为 Codex 在活跃任务中
临时改。WindowsApps/AppData 是否可移动应走对应应用/Windows 支持的迁移方式，不能把包目录手工剪切。

## 6. 推荐动作

### 现在

1. 保持现状；不要删除、改名或手工搬 `C:\Users\86136\.codex`。
2. 继续把模型、数据、checkpoint 留在服务器；Windows 只保留轻量文档、diff、日志摘要和必要视频。
3. 新的大型本地 repo/download 优先直接建在 E:；本次小 ZIP 留在当前 C: 工作区即可。
4. 关注 C: 空闲量；当前 37.99 GiB 尚可，但不再向 C: 下载 GB 级模型/数据。

### 若要彻底闭合 C -> E 迁移

把它作为单独维护任务：完全退出 ChatGPT/Codex/extension host/相关浏览器 native host，先盘点并备份，
把 SQLite 主库与 WAL/SHM 成组处理，统一 E: state，重新生成并验证 native-host/plugin 注册与 ACL，
确认新进程所有路径都指向 E: 后，才讨论 C: 旧目录的回收。

这一过程不能在当前 Codex 正运行、SQLite 正写入时完成，也不能只靠移动一个目录或创建 junction 就宣布
成功。
