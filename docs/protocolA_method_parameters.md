# Protocol A V13/K60 方法参数

本文件记录 GitHub 归档版采用的最终 V13/finally3 实验配置。真实标签不参与特征提取、降维、聚类、簇对齐或拒识，只在全部结果固定后用于多对一事后类别映射和外部聚类评价。

## 数据集

| 代码目录 | 论文名称 | 说明 |
|---|---|---|
| `F_new` | PV-Fruit | PlantVillage 水果病害子集 |
| `V_new` | PV-Vegetable | PlantVillage 蔬菜病害子集 |
| `M_new` | MCLD-11 | SHA256 去重后的完整 11 类多作物病害数据集 |
| `G_new` | DFLD-BR-4 | 去背景作物种类辅助数据集 |


## 特征提取

| 项目 | 参数 |
|---|---|
| 视觉骨干 | `timm.create_model("convnext_xlarge_in22k", pretrained=False, num_classes=0, global_pool="avg")` |
| 权重 | `models/model.safetensors` |
| 权重来源 | 公开 timm ConvNeXt-XLarge ImageNet-22K 权重 |
| 训练方式 | 本研究不微调；全部参数冻结，仅作为特征提取器 |
| 输入尺寸 | `224 x 224` |
| 颜色空间 | RGB |
| 预处理 | `Resize((224, 224))`，`ToTensor()` |
| 归一化 | mean `[0.485, 0.456, 0.406]`，std `[0.229, 0.224, 0.225]` |
| 特征层 | 全局平均池化输出 |
| 输出维度 | 2048 维 `float32` |

## 降维

| 项目 | 参数 |
|---|---|
| 主方法 | UMAP |
| `n_components` | 100 |
| `n_neighbors` | 15 |
| `min_dist` | 0.1 |
| `metric` | `euclidean` |
| `random_state` | 当前随机种子 |
| 随机种子 | `11, 22, 33, 44, 55` |

公平降维对照包括 Raw 2048 维、PCA 100 维和 UMAP 100 维。

## 聚类

| 算法 | 参数 |
|---|---|
| KMeans | `n_clusters=60`, `n_init=10`, `init="k-means++"`, `max_iter=300`, `tol=1e-4`, `random_state=seed` |
| Birch | `threshold=0.11`, `branching_factor=25`, `n_clusters=60` |
| Agglomerative | `n_clusters=60`, `linkage="ward"`，对应欧氏距离 |

## 跨算法簇对齐与共识拒识

| 步骤 | 定义 |
|---|---|
| 参考聚类 | 以 KMeans 簇编号作为参考 |
| 重叠矩阵 | 对 KMeans 簇 `i` 和另一算法簇 `j`，统计共同包含的样本数，形成 `60 x 60` overlap matrix |
| Hungarian 对齐 | 使用 `scipy.optimize.linear_sum_assignment(-overlap)` 最大化样本重叠 |
| 未匹配簇 | 映射不到参考簇时标记为 `-1` |
| any2 / C2 | 三种聚类结果中任意两个算法对齐后的簇编号一致则保留 |
| all3 / C3 | KMeans、Birch、Agglomerative 三者对齐后的簇编号全部一致则保留 |
| 拒识样本 | 不满足共识条件的样本标记为 Rejected，不参与 `acc_kept` 计算 |

## 事后评价

每个簇在聚类和拒识完成后映射到其保留样本中数量最多的真实类别：

```text
m(j) = argmax_c |C_j ∩ Y_c|
```

核心指标包括：

```text
Coverage = N_kept / N
Acc_kept = N_correct,kept / N_kept
Conservative accuracy = Acc_kept x Coverage = N_correct,kept / N
```

保守全样本准确率把拒识样本计入全样本分母，但不表示被拒识样本真实上都被错误分类。
