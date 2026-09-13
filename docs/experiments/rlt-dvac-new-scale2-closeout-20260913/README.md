# RLT DVAC new scale2 收尾

用户主动停止于采集 R475，原计划600轮；最近checkpoint目录R475。

[完整原始证据ZIP](rlt-dvac-new-scale2-R475-raw-evidence-20260913.zip)包含全部driver日志、TensorBoard events、逐轮指标CSV/JSON、实配、源码身份/文件SHA和checkpoint目录。CRC及逐文件SHA已验证。模型和回放不入包，仍在服务器。4张图及停止回执在旁列文件。

云端原始ZIP SHA256：`4bbbe92929758e11d37440c3e7546e638429aa313137b709596a21e9c831e245`。直接复用服务器归档，未往返重传。

用户本地ZIP `rlt-dvac-new-scale2-single-gpu600-R475-closeout-20260913.zip`另外加入4图/停止回执，SHA256为`2201329ca7cce65dff0a113bf0f5ad1acf9a54f0ce14ddb33e3649151e0439b9`。两包内容范围相容，但字节和SHA不同。

[四张单列图](plots/index.html) · [终点与逐tag摘要](SUMMARY.json) · [源码身份](source-identity.json) · [停止回执](stop-receipt.json)

云端将大日志、完整CSV/JSON和源码SHA清单保存在ZIP内，避免重复提交同份数据。前段采集来自Teacher；固定评估一直使用Student。
