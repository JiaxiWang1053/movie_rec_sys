import torch
import torch.nn as nn


class DeepFMRanker(nn.Module):
    """
    DeepFM 排序模型（稳定版）

    输出 logits，不做 sigmoid
    loss 端使用 BCEWithLogitsLoss
    """

    def __init__(
        self,
        sparse_feature_info: dict,
        num_genres: int,
        num_dense_features: int,
        embedding_dim: int = 16,
        hidden_dims=(128, 64),
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # ===== 一阶线性部分（FM first-order）=====
        self.first_order_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, 1)
            for feat, vocab_size in sparse_feature_info.items()
        })
        self.genre_first_order = nn.Embedding(num_genres, 1)

        # ===== 二阶 embedding（FM second-order + Deep input）=====
        self.feature_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embedding_dim)
            for feat, vocab_size in sparse_feature_info.items()
        })
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)

        # ===== dense 特征线性层 =====
        self.dense_linear = nn.Linear(num_dense_features, 1)

        # ===== Deep 部分 =====
        # sparse fields:
        # user_id, item_id, user_gender, user_age, user_occupation, item_genres_pooled
        num_sparse_fields = len(sparse_feature_info) + 1
        deep_input_dim = num_sparse_fields * embedding_dim + num_dense_features

        layers = []
        input_dim = deep_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h

        self.deep_layers = nn.Sequential(*layers)
        self.deep_output = nn.Linear(input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in self.first_order_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

        for emb in self.feature_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

        nn.init.xavier_uniform_(self.genre_first_order.weight)
        nn.init.xavier_uniform_(self.genre_embedding.weight)

        nn.init.xavier_uniform_(self.dense_linear.weight)
        nn.init.zeros_(self.dense_linear.bias)

        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.deep_output.weight)
        nn.init.zeros_(self.deep_output.bias)

    def pool_genre_embedding(self, genre_ids, genre_mask, embedding_layer):
        genre_emb = embedding_layer(genre_ids)            # [B, G, D] or [B, G, 1]
        genre_mask = genre_mask.unsqueeze(-1)             # [B, G, 1]
        masked = genre_emb * genre_mask

        denom = torch.clamp(genre_mask.sum(dim=1), min=1.0)
        pooled = masked.sum(dim=1) / denom
        return pooled

    def fm_first_order(
        self,
        user_id, item_id, user_gender, user_age, user_occupation,
        item_genre_ids, item_genre_mask,
        dense_features
    ):
        first_order_sum = (
            self.first_order_embeddings["user_id"](user_id) +
            self.first_order_embeddings["item_id"](item_id) +
            self.first_order_embeddings["user_gender"](user_gender) +
            self.first_order_embeddings["user_age"](user_age) +
            self.first_order_embeddings["user_occupation"](user_occupation)
        )  # [B, 1]

        genre_first = self.pool_genre_embedding(
            item_genre_ids, item_genre_mask, self.genre_first_order
        )  # [B, 1]

        dense_first = self.dense_linear(dense_features)  # [B, 1]

        return first_order_sum + genre_first + dense_first

    def fm_second_order(
        self,
        user_id, item_id, user_gender, user_age, user_occupation,
        item_genre_ids, item_genre_mask
    ):
        emb_list = [
            self.feature_embeddings["user_id"](user_id),
            self.feature_embeddings["item_id"](item_id),
            self.feature_embeddings["user_gender"](user_gender),
            self.feature_embeddings["user_age"](user_age),
            self.feature_embeddings["user_occupation"](user_occupation),
            self.pool_genre_embedding(
                item_genre_ids, item_genre_mask, self.genre_embedding
            ),
        ]

        x = torch.stack(emb_list, dim=1)  # [B, F, D]

        sum_square = torch.sum(x, dim=1) ** 2
        square_sum = torch.sum(x ** 2, dim=1)
        second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        return second_order, x

    def deep_part(self, sparse_emb_tensor, dense_features):
        batch_size = sparse_emb_tensor.size(0)
        sparse_flat = sparse_emb_tensor.reshape(batch_size, -1)
        deep_input = torch.cat([sparse_flat, dense_features], dim=1)

        hidden = self.deep_layers(deep_input)
        out = self.deep_output(hidden)  # [B,1]
        return out

    def forward(
        self,
        user_id, item_id, user_gender, user_age, user_occupation,
        item_genre_ids, item_genre_mask,
        dense_features
    ):
        first_order = self.fm_first_order(
            user_id, item_id, user_gender, user_age, user_occupation,
            item_genre_ids, item_genre_mask,
            dense_features
        )  # [B,1]

        second_order, sparse_emb_tensor = self.fm_second_order(
            user_id, item_id, user_gender, user_age, user_occupation,
            item_genre_ids, item_genre_mask
        )  # [B,1], [B,F,D]

        deep_out = self.deep_part(sparse_emb_tensor, dense_features)  # [B,1]

        logits = first_order + second_order + deep_out   # [B,1]
        return logits.squeeze(1)  # [B]