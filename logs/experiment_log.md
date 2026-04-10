# 推荐系统项目实验日志（MovieLens 1M）

---


# 实验 1：ItemCF baseline

## 模型

* Item-based Collaborative Filtering（ItemCF）
* 相似度计算：

  * 共现次数归一化：
    sim(i,j) = co_occurrence(i,j) / sqrt(cnt(i) * cnt(j))

## 数据处理

* 使用正样本（rating ≥ 4）
* 用户历史按时间排序
* 划分：

  * train：历史行为
  * valid：倒数第2个
  * test：最后一个

## 评估方式

* 候选集：

  * 1 个 target_item（valid）
  * 99 个随机负样本
* 指标：

  * HitRate@5 / @10 / @20
  * Recall@K（此任务中等同 HitRate）

## 结果

* HitRate@5: 0.455178
* HitRate@10: 0.624027
* HitRate@20: 0.797680

## 结论

* ItemCF 在 MovieLens 上表现较强
* 利用用户-物品共现关系，可以很好捕捉协同过滤信号
* 作为后续模型对比的 baseline

---

# 实验 2：基础双塔（ID-only Two-Tower）

## 模型

* 用户塔：

  * user_id embedding
* 物品塔：

  * item_id embedding
* 打分：

  * dot product（点积）
* 损失函数：

  * BPR Loss

## 训练数据

* (user_id, pos_item, neg_item)
* 负样本：

  * 从未交互物品中随机采样

## 超参数

* embedding_dim = 64
* batch_size = 1024
* learning_rate = 1e-3
* epoch = 3

## 评估方式

与 ItemCF 完全一致：

* 相同候选集
* 相同指标（HitRate@K）

## 结果

* HitRate@5: 0.394200
* HitRate@10: 0.573322
* HitRate@20: 0.757415

## 结论

* 表现弱于 ItemCF baseline
* 原因分析：

  * 仅使用 ID embedding，表达能力有限
  * 随机负样本过于简单
  * 模型难以学习细粒度偏好
* 说明：

  * 神经模型并非天然优于传统方法
  * baseline 是必要的

---

# 实验 3：双塔 + 用户历史行为（History Pooling）

## 模型改进

在用户塔中加入历史行为：

用户向量：

u = user_id_embedding + mean(history_item_embeddings)

即：

* user_id embedding（个体偏好）
* * 历史电影 embedding 平均（兴趣表达）

## 输入数据

训练样本变为：

(user_id, history_items, pos_item, neg_item)

其中：

* history_items = 当前正样本之前的训练历史

## 训练方式

* 同样使用 BPR Loss
* 负样本仍为随机采样
* 使用 padding + mask 处理变长历史

## 超参数

* embedding_dim = 64
* batch_size = 512
* learning_rate = 1e-3
* epoch = 3

## 模型保存

* 使用 checkpoint 保存模型：

  * checkpoints/two_tower_history.pth

## 评估方式

* 使用 train history 作为用户输入
* 候选集同 baseline（1 正样本 + 99 负样本）

## 结果

HitRate@5: 0.433637
Recall@5: 0.433637
HitRate@10: 0.614913
Recall@10: 0.614913
HitRate@20: 0.793703
Recall@20: 0.793703

## 结论（预期）

* 引入用户历史后，模型能够学习用户兴趣分布
* 相比 ID-only 模型，表达能力明显增强
* 通常会带来 HitRate 提升

---

# 当前总结

当前完成的召回模块：

1. ItemCF baseline
2. Two-Tower（ID only）
3. Two-Tower + History

观察：

* ItemCF 当前最强
* 基础双塔不足
* 引入历史是关键改进方向

---

# 实验 4：双塔 + History + Hard Negative（ItemCF-based）

## 模型改进

在 “Two-Tower + History” 的基础上，引入困难负样本（Hard Negative）：

* 使用 ItemCF 为每个用户生成候选集合（TopK）
* 去除用户已看过电影
* 剩余部分作为 hard negative pool

训练时：

* 负样本优先从 hard negative pool 采样
* 不足部分由随机负样本补齐

---

## 负样本策略

* Hard Negative 来源：

  * ItemCF 推荐结果（TopK=200）
* Random Negative 来源：

  * 未交互电影集合随机采样
* 当前策略：

  * 每个正样本配 1 个负样本
  * 优先 hard negative

---

## 超参数

* embedding_dim = 64
* batch_size = 512
* learning_rate = 1e-3
* epoch = 3
* hard_negative_topk = 200
* num_negatives_per_positive = 1

---

## 结果

* HitRate@5: 0.131897
* HitRate@10: 0.201326
* HitRate@20: 0.291632

---

## 结论

* 引入纯 hard negative 后，召回性能显著下降（大幅低于 history 版本）

* 推测原因：

  * 负样本难度过高，训练分布发生严重偏移
  * 模型被迫区分过于相似的样本，缺乏基础区分能力
  * 缺少 easy negative（随机负样本）导致训练不稳定

* 说明：

  * hard negative 并非越难越好
  * 负样本分布需要平衡（easy vs hard）
  * 直接使用 ItemCF TopK 作为负样本会破坏训练信号

---

## 问题分析（关键）

当前训练目标变为：

* pos vs 极难负样本（hard negative）

缺失：

* pos vs easy negative（明显不喜欢）

导致模型：

* 无法建立全局表示空间
* 只学习局部细粒度区分
* 排序能力整体下降

---

# 实验 5：双塔 + History + Mixed Negative

## 模型改进

在 “Two-Tower + History” 的基础上，将纯 hard negative 改为混合负样本策略（Mixed Negative Sampling）：

* 每个正样本生成两条训练样本：

  * 1 条使用 hard negative
  * 1 条使用 random negative
* 其中：

  * hard negative 来自 ItemCF 候选池
  * random negative 来自未交互物品集合随机采样

相较于实验 4，本实验的目标是：

* 保留 hard negative 的细粒度区分能力
* 同时利用 random negative 保持全局区分能力
* 缓解纯 hard negative 导致的训练分布失衡问题

---

## 负样本策略

* Hard Negative 来源：

  * ItemCF 推荐结果（TopK=200）
* Random Negative 来源：

  * 未交互电影集合随机采样
* 当前策略：

  * 每个正样本配 2 条训练样本：

    * 1 个 hard negative
    * 1 个 random negative

---

## 超参数

* embedding_dim = 64
* batch_size = 512
* learning_rate = 1e-3
* epoch = 3
* hard_negative_topk = 200

---

## 结果

* HitRate@5: 0.417233
* HitRate@10: 0.593869
* HitRate@20: 0.795360

---

## 结论

* 相比纯 hard negative，Mixed Negative 显著恢复了模型性能：

  * HitRate@10 从 0.201 提升到 0.594
* 说明：

  * 纯 hard negative 导致训练分布过于极端
  * 引入 random negative 后，训练稳定性明显改善
* 但相比 “Two-Tower + History”：

  * HitRate@10 从 0.615 略降至 0.594
* 说明当前 hard negative 设计仍未带来真正收益

---

## 问题分析

可能原因包括：

* 当前 hard negative 候选池仍不够精准
* hard negative 与 random negative 的比例仍不理想
* 当前双塔模型容量较弱，仅使用：

  * user_id embedding
  * history mean pooling
  * item_id embedding

因此模型虽然能从 mixed negative 中恢复训练稳定性，但尚不足以充分利用更复杂的负样本信息。

---


# 实验 6：双塔 + History + MLP + Mixed Negative（最终版本）

## 模型改进

在前面实验的基础上，组合所有优化策略：

* History pooling（用户兴趣建模）
* MLP（增强非线性表达）
* Mixed Negative（平衡负样本分布）

即：

Two-Tower + History + MLP + Mixed Negative

---

## 训练方式

* 负样本策略：

  * Mixed Negative（1 hard + 1 random）
* 损失函数：

  * BPR Loss

---

## 超参数

* embedding_dim = 64
* hidden_dim = 128
* batch_size = 512
* learning_rate = 1e-3
* epoch = 3

---

## 结果

* HitRate@5: 0.390389
* HitRate@10: 0.576968
* HitRate@20: 0.781276

---

## 结论

* 最终组合模型性能低于：

  * Two-Tower + History（0.615）
  * Mixed Negative（0.594）

* 说明：

  * 简单叠加优化策略并不能保证性能提升
  * 模型复杂度提升未带来有效信息增益

---

## 分析

* 当前任务中：

  * 用户历史行为（history）是最关键特征
  * 简单 mean pooling 已能有效表达用户兴趣

* MLP：

  * 未引入新特征，仅增加模型复杂度
  * 在当前数据规模下未带来收益

* Mixed Negative：

  * 在当前模型能力下未提供有效提升
  * 可能引入额外噪声

---

#
# 实验 9：双塔 + History + Dynamic Hard Negative

## 模型改进

在当前最佳基础模型 “Two-Tower + History” 上，引入动态困难负样本（Dynamic Hard Negative）。

与静态 hard negative 不同，本实验中的 hard negative 不是固定由 ItemCF 或其他启发式方法一次性生成，而是：

* 在每个 epoch 开始前
* 使用当前双塔模型对用户未交互电影进行打分
* 选取当前模型最容易误判为正样本的高分未交互电影
* 作为该 epoch 的 hard negative pool

即：

* hard negative 会随着模型参数更新而动态变化
* 模型始终在学习“当前最难区分的负样本”

---

## 负样本策略

每个 epoch 动态更新一次 hard negative pool：

1. 对每个用户，从未交互电影中随机采样一批候选（candidate sample）
2. 用当前模型对候选物品打分
3. 取分数最高的若干个作为 dynamic hard negatives
4. 训练时使用：

   * dynamic hard negative
   * random negative

即每个正样本生成两条训练样本：

* 1 条 hard negative
* 1 条 random negative

---

## 超参数

* embedding_dim = 64
* batch_size = 512
* learning_rate = 1e-3
* epoch = 3
* hard_pool_size = 50
* candidate_sample_size = 300

---

## 结果

* HitRate@5: 0.553438
* HitRate@10: 0.726263
* HitRate@20: 0.870920

---

## 结论

* 相比 “Two-Tower + History”：

  * HitRate@10 从 0.615 提升到 0.726

* 相比 ItemCF baseline：

  * HitRate@10 从 0.624 提升到 0.726

说明动态困难负样本显著提升了双塔召回能力，并成功超过传统协同过滤 baseline。

---

## 分析

* 与静态 hard negative 相比，dynamic hard negative 能更准确地对准“当前模型最容易犯错的负样本”
* 与 random negative 相比，dynamic hard negative 提供了更强的细粒度学习信号
* 这说明当前召回性能瓶颈主要在负样本质量，而不是单纯增加模型复杂度

---

## 当前阶段结论

在 MovieLens 1M 上，当前最佳召回模型为：

**Two-Tower + History + Dynamic Hard Negative**

该模型首次显著超过 ItemCF baseline，是当前推荐系统项目召回阶段的最终最优版本。



# 实验 X：多正样本评估（Multi-Positive Evaluation）

## 背景

原始召回评估采用 leave-one-out 方式：

* 每个用户仅包含 1 个正样本
* 导致：

  * HitRate@K = Recall@K

该评估方式不能真实反映召回模型在实际推荐场景中的表现。

---

## 改进评估方式

采用多正样本评估协议：

* 对每个用户：

  * train：历史行为
  * test：最后 n=5 个正样本
* 候选集：

  * 多个正样本（target_items）
  * * 100 个随机负样本

---

## 指标定义

* HitRate@K：

  * TopK 是否命中至少 1 个正样本

* Recall@K：

  * 命中正样本数 / 总正样本数

* NDCG@K：

  * 命中位置的排序质量

---

## 对比模型

1. ItemCF（传统协同过滤）
2. TwoTower + History + Dynamic Hard Negative（当前最优召回）

---

## 结果对比

### ItemCF

* HitRate@5: 0.841799

* Recall@5: 0.362598

* NDCG@5: 0.384270

* HitRate@10: 0.940716

* Recall@10: 0.557169

* NDCG@10: 0.490893

* HitRate@20: 0.986178

* Recall@20: 0.760699

* NDCG@20: 0.578360

---

### TwoTower + History + Dynamic Hard Negative

* HitRate@5: 0.927227

* Recall@5: 0.456020

* NDCG@5: 0.482053

* HitRate@10: 0.981349

* Recall@10: 0.677002

* NDCG@10: 0.603443

* HitRate@20: 0.998668

* Recall@20: 0.856719

* NDCG@20: 0.681042

---

## 提升分析

以 K=10 为例：

* Recall@10：

  * 0.557 → 0.677（+21.5%）

* NDCG@10：

  * 0.491 → 0.603（+22.8%）

* HitRate@10：

  * 0.941 → 0.981（+4%）

---

## 结论

* 在更真实的多正样本评估协议下：

  * 双塔模型在所有指标上均显著优于 ItemCF

* Dynamic Hard Negative 的效果稳定：

  * 不仅在单正样本评估下有效
  * 在多正样本场景下仍然带来显著提升

* 模型能力提升体现在：

  * 能找回更多真实相关物品（Recall ↑）
  * 能将相关物品排在更前（NDCG ↑）

---

## 关键结论（总结）

* 传统 ItemCF 依赖共现统计，表达能力有限
* 双塔模型通过 embedding 学习用户兴趣表示
* 动态困难负样本训练显著增强模型区分能力


