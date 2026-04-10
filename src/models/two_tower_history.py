import torch
import torch.nn as nn


class TwoTowerHistoryRetrievalModel(nn.Module):
    """
    带用户历史 pooling 的双塔召回模型。
    用户向量 = user_id embedding + 历史电影 embedding 的平均池化
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def encode_user(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        用户编码：
        user_id_emb + mean(history_item_emb)
        """
        user_id_emb = self.user_embedding(user_ids)  # [B, D]

        history_emb = self.item_embedding(history_item_ids)   # [B, L, D]
        history_mask = history_mask.unsqueeze(-1)             # [B, L, 1]

        masked_history_emb = history_emb * history_mask

        # 防止除 0
        history_len = torch.clamp(history_mask.sum(dim=1), min=1.0)   # [B, 1]
        history_mean_emb = masked_history_emb.sum(dim=1) / history_len # [B, D]

        user_vec = user_id_emb + history_mean_emb
        return user_vec

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids)

    def score(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_vec = self.encode_user(user_ids, history_item_ids, history_mask)
        item_vec = self.encode_item(item_ids)
        scores = torch.sum(user_vec * item_vec, dim=1)
        return scores

    def forward(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ):
        pos_scores = self.score(
            user_ids, history_item_ids, history_mask, pos_item_ids
        )
        neg_scores = self.score(
            user_ids, history_item_ids, history_mask, neg_item_ids
        )
        return pos_scores, neg_scores


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12)
    return loss.mean()