import torch
import torch.nn as nn


class TwoTowerRetrievalModel(nn.Module):
    """
    最基础版本的双塔召回模型。

    当前版本只使用：
    - user_id embedding
    - item_id embedding

    后续可以逐步扩展：
    - 用户属性特征
    - 电影属性特征
    - 用户历史行为 pooling
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64
    ):
        super().__init__()

        # 用户塔：用户ID Embedding
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )

        # 物品塔：电影ID Embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )

        # 参数初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        输入用户ID，输出用户向量
        """
        return self.user_embedding(user_ids)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        输入物品ID，输出物品向量
        """
        return self.item_embedding(item_ids)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        计算 user-item 匹配分数（点积）
        输出 shape: [batch_size]
        """
        user_vec = self.encode_user(user_ids)   # [B, D]
        item_vec = self.encode_item(item_ids)   # [B, D]

        # 按 embedding 维度做点积
        scores = torch.sum(user_vec * item_vec, dim=1)
        return scores

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ):
        """
        前向传播：
        - 计算正样本分数
        - 计算负样本分数

        返回：
        - pos_scores: [batch_size]
        - neg_scores: [batch_size]
        """
        pos_scores = self.score(user_ids, pos_item_ids)
        neg_scores = self.score(user_ids, neg_item_ids)

        return pos_scores, neg_scores


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    BPR Loss:
        -log(sigmoid(pos_score - neg_score))

    目标：
    让正样本分数高于负样本分数
    """
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12)
    return loss.mean()