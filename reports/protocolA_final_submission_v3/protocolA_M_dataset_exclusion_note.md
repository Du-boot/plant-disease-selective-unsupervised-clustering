# M_new 与 M_new_drop5_drop7 类别排除说明

根据当前服务器文件统计：

完整 `M_new` 共 11 类：

| 完整M类号 | 类别 | 样本数 |
|---:|---|---:|
| 0 | Cashew - leaf miner | 555 |
| 1 | Cashew - red rust | 566 |
| 2 | Corn - leaf blight | 493 |
| 3 | Corn - streak virus | 401 |
| 4 | Potato - fungi | 390 |
| 5 | Potato - nematode | 68 |
| 6 | Rice - bacterial leaf blight | 401 |
| 7 | Rice - brown spot | 370 |
| 8 | Rice - leaf blast | 334 |
| 9 | Tomato - septoria leaf spot | 720 |
| 10 | Tomato - verticillium wilt | 673 |

当前主实验使用的 `M_new_drop5_drop7` 共 9 类，文件前缀已经重新编号：

| 清洗版类号 | 对应完整M类号 | 类别 | 样本数 |
|---:|---:|---|---:|
| 0 | 0 | Cashew - leaf miner | 555 |
| 1 | 1 | Cashew - red rust | 566 |
| 2 | 2 | Corn - leaf blight | 493 |
| 3 | 3 | Corn - streak virus | 401 |
| 4 | 4 | Potato - fungi | 390 |
| 5 | 6 | Rice - bacterial leaf blight | 401 |
| 6 | 7 | Rice - brown spot | 370 |
| 7 | 9 | Tomato - septoria leaf spot | 720 |
| 8 | 10 | Tomato - verticillium wilt | 673 |

因此，从当前文件和构建脚本看，9类子集相对于完整M未纳入：

- 完整M类号5：`Potato - nematode`，样本数68；
- 完整M类号8：`Rice - leaf blast`，样本数334。

论文中不要只写“删除第5、7类”，因为清洗后类别已经重新编号，容易造成类号错位。建议写具体类别名。

当前更稳妥的表述：

> 为分析类别构成、任务范围和样本规模差异对无监督聚类结果的影响，本文同时报告完整11类 `M_new` 数据集和预先整理的9类子集。9类子集未纳入 `Potato - nematode` 和 `Rice - leaf blast` 两个类别，其中 `Potato - nematode` 样本量明显较少，且属于线虫危害，与典型叶部真菌、细菌或病毒病斑表型存在任务范围差异。对于 `Rice - leaf blast`，在缺乏额外质量异常证据时，本文不将其描述为错误或低质量类别，而仅将9类版本作为预先整理的子集分析。完整11类结果同步报告，用于验证结论在不排除类别情况下的稳健性。

需要注意：

> 不能写成“由于这两类准确率低所以删除”。完整M与清洗M应同时报告，以避免选择性删类质疑。
