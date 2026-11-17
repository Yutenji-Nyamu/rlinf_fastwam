# Interleaved batched K inference

Same N17 observation, H100, NFE10. Each K has three warmups and30 measurements in shuffled interleaved order.

| K | Median ms | Q25/Q75 ms | Peak allocated GiB |
|---|---:|---:|---:|
| 1 | 168.01 | 138.13/171.72 | 7.145 |
| 2 | 189.73 | 159.48/195.32 | 7.287 |
| 3 | 204.25 | 173.76/207.27 | 7.429 |
| 4 | 212.73 | 182.25/216.33 | 7.572 |
| 8 | 260.04 | 233.71/265.23 | 8.132 |

K2 toK3 increases here (189.73 to204.25 ms). K4 toK8 adds47.32 ms and0.560 GiB. Full-round overhead is collected separately.
