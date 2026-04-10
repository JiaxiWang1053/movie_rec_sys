# 推荐系统项目实验日志（排序阶段）

---

# 实验 1：LightGBM 排序模型（Baseline Ranker）

## 模型

* LightGBM 排序模型（LambdaRank）
* 目标函数：

  * `lambdarank`（直接优化排序指标，如 NDCG）

## 输入数据

来自召回阶段（TwoTower + Dynamic Hard Negative）：

* 每个用户：

  * 1 个正样本（valid/test）
  * 99 个负样本
* 展开为：

  * `(user, item, label)` 形式

## 特征构造

### 稀疏特征（离散）

* user_id
* item_id
* user_gender
* user_age
* user_occupation

### 稠密特征（连续）

* retrieval_score（召回模型输出）⭐
* user_history_len
* item_popularity
* genre_overlap_count
* item_year

## 训练方式

* group-by user（每个用户一个排序列表）
* LightGBM 自动学习特征分裂与非线性交互

## 评估方式

* 候选集：

  * 1 正样本 + 99 负样本
* 指标：

  * AUC
  * HitRate@K
  * NDCG@K

## 结果

* AUC: 0.907372
* HitRate@5: 0.543993
* NDCG@5: 0.387519
* HitRate@10: 0.713836
* NDCG@10: 0.442497
* HitRate@20: 0.860315
* NDCG@20: 0.479735

## 特征重要性分析

Top 特征：

1. item_popularity
2. user_history_len
3. retrieval_score
4. item_year

较弱特征：

* genre_overlap_count
* user_gender

## 结论

* LightGBM 能有效利用结构化特征进行排序
* retrieval_score 被显著利用，说明召回信号有效传递到排序阶段
* 模型较依赖统计特征（popularity、history），而非内容匹配
* 在当前数据规模和特征设计下，表现稳定且较强

---

# 实验 2：DeepFM 排序模型（第一版）

## 模型

* DeepFM（FM + Deep Neural Network）
* 结构：

  * FM：学习二阶特征交叉
  * Deep：学习高阶非线性关系
* 输出：

  * sigmoid(logit)
* 损失函数：

  * BCEWithLogitsLoss

## 输入数据

与 LightGBM 完全一致：

* 同一候选集（1 正 + 99 负）
* 同一标签构造

## 特征设计

### 稀疏特征（embedding）

* user_id
* item_id
* user_gender
* user_age
* user_occupation

### 多值稀疏特征

* item_genres（embedding + mean pooling）

### 稠密特征（dense）

* retrieval_score ⭐
* user_history_len
* item_popularity
* genre_overlap_count
* item_year

## 关键工程处理

* dense 特征标准化（mean/std）
* logits + BCEWithLogitsLoss（数值稳定）
* padding + mask 处理 genre 多值特征

## 训练设置

* embedding_dim = 16
* hidden_dims = (128, 64)
* batch_size = 512
* learning_rate = 1e-3
* epoch = 5（早停效果明显）

## 结果

* AUC: 0.906280
* HitRate@5: 0.533554
* NDCG@5: 0.376142
* HitRate@10: 0.706711
* NDCG@10: 0.431968
* HitRate@20: 0.859652
* NDCG@20: 0.470870

## 训练现象

* 第1轮 AUC 即达到最佳（0.906）
* 后续 epoch 出现下降（过拟合）

## 对比 LightGBM

| 指标         | LightGBM | DeepFM |
| ---------- | -------- | ------ |
| AUC        | 0.907    | 0.906  |
| HitRate@10 | 0.714    | 0.707  |
| NDCG@10    | 0.442    | 0.432  |

## 结论

* DeepFM 成功学习到有效排序能力（AUC 接近 LightGBM）
* 但整体略弱于 LightGBM
* 原因分析：

  * 当前强特征主要为 dense（retrieval_score、popularity）
  * DeepFM 优势（sparse 交叉）未完全发挥
  * 数据规模对深模型不够友好

---

# 当前排序阶段总结

当前排序模块对比：

1. LightGBM（最佳）
2. DeepFM（略弱）

核心观察：

* retrieval_score 是排序阶段最关键特征
* 统计特征（popularity / history）贡献显著
* 内容匹配特征（genre）仍较弱

关键结论：

> 在当前数据规模与特征结构下，基于特征工程的树模型（LightGBM）仍优于神经排序模型（DeepFM）。

---
