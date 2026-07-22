# Protocol A 方法参数核查表

本表根据服务器实际实验代码核实，代码位置主要包括：

- `/data1/D/hostal/audit_unsupervised_experiments.py`
- `/data1/D/hostal/protocolA_followup_experiments.py`
- `/data1/D/hostal/protocolA_final_supplements.py`

## 特征提取

| 模块 | 参数 |
|---|---|
| 视觉骨干 | ConvNeXt-XLarge结构，代码中使用 `timm.create_model("convnext_xlarge_in22k", pretrained=False, num_classes=0, global_pool="avg")` 初始化；timm运行时会将该旧名称映射到当前ConvNeXt-XLarge实现 |
| 权重 | `/data1/D/deploy/cn/model.safetensors`，作为本地外部权重检查点载入 |
| 权重来源 | 当前文件无safetensors metadata，目录中尚未找到明确下载仓库、许可证或训练来源记录；投稿前应补充权重最初来源。来源确认前，正文不应写成“使用ImageNet-22K官方预训练权重” |
| 权重载入 | 去掉 `model.` 前缀，排除 `head.weight` 和 `head.bias`，`strict=False` |
| 是否微调 | 不微调；`model.eval()`，`torch.no_grad()`，仅作冻结特征提取器 |
| 输入尺寸 | `224 × 224` |
| 图像颜色 | RGB |
| 预处理 | `Resize((224, 224))`，`ToTensor()` |
| 归一化 | mean = `[0.485, 0.456, 0.406]`，std = `[0.229, 0.224, 0.225]` |
| 池化层 | `global_pool="avg"` |
| 输出特征 | 2048维视觉特征，`float32` |

## 降维

| 模块 | 参数 |
|---|---|
| 主方法 | UMAP |
| 主配置 | `n_components=100` |
| `n_neighbors` | 15 |
| `min_dist` | 0.1 |
| `metric` | `euclidean` |
| `random_state` | 当前随机种子，主实验为 `11, 22, 33, 44, 55` |
| 对照 | PCA 100维；Raw 2048维 |

## 聚类

| 算法 | 参数 |
|---|---|
| KMeans | `n_clusters=20`, `n_init=10`, `random_state=seed` |
| KMeans未显式指定项 | `init="k-means++"`, `max_iter=300`, `tol=1e-4` 使用 scikit-learn 默认值 |
| Birch | `threshold=0.11`, `branching_factor=25`, `n_clusters=20` |
| Agglomerative | `n_clusters=20`, `linkage="ward"`；Ward linkage对应欧氏距离 |

## 簇编号对齐与共识拒识

| 步骤 | 定义 |
|---|---|
| 参考聚类 | KMeans作为参考聚类 |
| 重叠矩阵 | 对KMeans簇 `i` 和另一算法簇 `j`，统计二者共同包含的样本数，得到 `20 × 20` overlap matrix |
| 簇对齐 | 使用 `scipy.optimize.linear_sum_assignment(-overlap)`，即Hungarian算法最大化样本重叠 |
| 未匹配簇 | 映射不到参考簇时标记为 `-1` |
| any2 | 三个聚类结果中任意两个算法对齐后的簇编号一致，则保留 |
| all3 | KMeans、Birch、Agglomerative三个算法对齐后的簇编号全部一致，则保留 |
| 拒识 | 不满足共识条件的样本标记为Rejected，不参与保留样本 `acc_kept` |
| 评价映射 | 聚类、对齐和拒识全部固定后，真实标签仅用于多对一事后类别映射和外部评价 |

## 数据审计口径

| 项目 | 处理 |
|---|---|
| 完全重复 | 使用SHA256识别字节级完全重复图像，并删除重复副本 |
| pHash近重复 | 仅作为近重复风险审计；不据此删图，不参与特征提取、降维、聚类、拒识或评价 |

## 主实验配置

| 项目 | 值 |
|---|---|
| UMAP维度 | 100 |
| 聚类簇数 | 20 |
| 随机种子 | 11, 22, 33, 44, 55 |
| 主指标 | `acc_kept`，即高置信保留样本的事后对齐聚类准确率 |
| 辅助指标 | rejection rate / coverage / overall accuracy / ARI / NMI / AMI / Homogeneity / Completeness / V-measure |
