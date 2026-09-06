# Evidence 目录规则

这里只保存可快速复核的小型证据，不保存源码副本、模型、数据集、大日志或认证信息。

计划保存：

- 服务器只读 Git `HEAD/status/origin/diffstat`。
- standalone Fast-WAM resolved config 摘要与黄金fixture：`fastwam-golden/adjust_bottle-official-45d8e145/{fixture.pt,metadata.json}`。不含模型权重，测试用显式路径或环境变量读取。
- 社区分支和本项目功能分支的 `diff --stat`/文件列表。
- ratio parity、参数 hash、checkpoint resume/export 的小型测试摘要。

动态训练指标仍存放在项目现有 `audits/` 与权威交接中。

## 2026-07-17 公开版本核查

```text
RLinf server pin from migration record:
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf

git ls-remote official tag v0.3:
  0505431899574619da86f551bad70b71e0ea2177

git ls-remote official main:
  c5ca51cc21c007a41d287159f9e1b14e0200000e

server pin -> current main:
  c90951a  fix SFT co-training loss application
  c5ca51c  add v0.3 release notes
```

`v0.3` tag 与 main 是分叉/cherry-pick 历史，不能把 tag 当成 server pin 的普通 fast-forward 后继。
