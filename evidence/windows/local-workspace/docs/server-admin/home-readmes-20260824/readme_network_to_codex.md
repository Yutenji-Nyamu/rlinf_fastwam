# 服务器网络说明

> 以下是使用建议，不构成强制规范，具体以人类判断和实时状态为准。

- 服务器运行共享 Mihomo，监听 `127.0.0.1:7890`。新登录的 Shell 会自动设置代理变量；Git、curl、pip、Conda、Hugging Face 等支持这些变量的程序会先进入 Mihomo，再由规则决定直连或代理。
- 当前规则让本机、常用国内站点、ModelScope、Gitee、PyPI、Conda、arXiv 和 W&B 等直接连接；GitHub、Hugging Face、Docker Hub、Google、OpenAI/ChatGPT 等经 `AUTO`，未匹配站点也由 `AUTO` 处理。
- `DIRECT` 不消耗订阅流量；`AUTO` 每隔一段时间检测节点并自动选择当前可用且较快的节点。它不是 TUN，不读取代理变量的程序不会自动进入 Mihomo。
- 可用 `env | grep -i '_proxy='` 查看当前 Shell；用 `curl -I https://目标网址` 按现行规则测试，或用 `curl --noproxy '*' -I https://目标网址` 强制绕过 Mihomo 直连。
- 共享代理由管理员统一维护，普通用户通常无需复制软件或订阅配置。
