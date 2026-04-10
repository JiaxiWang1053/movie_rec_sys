import torch
import torch.nn as nn


class TwoTowerHistoryUserFeatRetrievalModel(nn.Module):
    """
    双塔召回模型（History + User Features）

    用户塔输入：
    - user_id embedding
    - history item embedding mean pooling
    - gender embedding
    - age embedding
    - occupation embedding

    然后拼接后送入 MLP，得到最终用户向量。

    物品塔输入：
    - item_id embedding
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_gender: int,
        num_age: int,
        num_occupation: int,
        gender_ids: torch.Tensor,
        age_ids: torch.Tensor,
        occupation_ids: torch.Tensor,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        user_feat_dim: int = 16
    ):
        super().__init__()

        # ===== ID Embedding =====
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )

        # ===== 用户侧离散特征 Embedding =====
        self.gender_embedding = nn.Embedding(
            num_embeddings=num_gender,
            embedding_dim=user_feat_dim
        )

        self.age_embedding = nn.Embedding(
            num_embeddings=num_age,
            embedding_dim=user_feat_dim
        )

        self.occupation_embedding = nn.Embedding(
            num_embeddings=num_occupation,
            embedding_dim=user_feat_dim
        )

        # ===== 用户塔 MLP =====
        user_input_dim = embedding_dim * 2 + user_feat_dim * 3

        self.user_mlp = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # ===== 参数初始化 =====
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.gender_embedding.weight)
        nn.init.xavier_uniform_(self.age_embedding.weight)
        nn.init.xavier_uniform_(self.occupation_embedding.weight)

        for layer in self.user_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # ===== 注册用户特征索引表 =====
        # 这样 forward 时可以直接用 user_id 查对应特征 id
        self.register_buffer("gender_ids", gender_ids)
        self.register_buffer("age_ids", age_ids)
        self.register_buffer("occupation_ids", occupation_ids)

    def encode_user(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        用户编码：
        user_id_emb + history_mean_emb + demographic feature embeddings
        然后拼接后送入 MLP
        """
        # ===== 1. user_id embedding =====
        user_id_emb = self.user_embedding(user_ids)  # [B, D]

        # ===== 2. history mean pooling =====
        history_emb = self.item_embedding(history_item_ids)   # [B, L, D]
        history_mask = history_mask.unsqueeze(-1)             # [B, L, 1]

        masked_history_emb = history_emb * history_mask
        history_len = torch.clamp(history_mask.sum(dim=1), min=1.0)   # [B, 1]
        history_mean_emb = masked_history_emb.sum(dim=1) / history_len # [B, D]

        # ===== 3. 用户侧静态特征 =====
        gender_feat_ids = self.gender_ids[user_ids]               # [B]
        age_feat_ids = self.age_ids[user_ids]                     # [B]
        occupation_feat_ids = self.occupation_ids[user_ids]       # [B]

        gender_emb = self.gender_embedding(gender_feat_ids)       # [B, F]
        age_emb = self.age_embedding(age_feat_ids)                # [B, F]
        occupation_emb = self.occupation_embedding(occupation_feat_ids)  # [B, F]

        # ===== 4. 拼接送入 MLP =====
        user_input = torch.cat(
            [user_id_emb, history_mean_emb, gender_emb, age_emb, occupation_emb],
            dim=1
        )

        user_vec = self.user_mlp(user_input)   # [B, D]
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
            user_ids=user_ids,
            history_item_ids=history_item_ids,
            history_mask=history_mask,
            item_ids=pos_item_ids
        )

        neg_scores = self.score(
            user_ids=user_ids,
            history_item_ids=history_item_ids,
            history_mask=history_mask,
            item_ids=neg_item_ids
        )

        return pos_scores, neg_scores


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12)
    return loss.mean()