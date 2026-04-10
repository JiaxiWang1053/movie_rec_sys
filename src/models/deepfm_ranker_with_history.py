import torch
import torch.nn as nn


class DeepFMRankerWithHistory(nn.Module):
    """
    DeepFM + 用户历史 pooling
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

        self.first_order_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, 1)
            for feat, vocab_size in sparse_feature_info.items()
        })
        self.genre_first_order = nn.Embedding(num_genres, 1)

        self.feature_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embedding_dim)
            for feat, vocab_size in sparse_feature_info.items()
        })
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)

        # history 复用 item_id embedding
        self.dense_linear = nn.Linear(num_dense_features, 1)

        # fields:
        # user_id, item_id, user_gender, user_age, user_occupation, item_genre_pooled, history_pooled
        num_sparse_fields = len(sparse_feature_info) + 2
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

    def pool_embedding(self, ids, mask, embedding_layer):
        emb = embedding_layer(ids)
        mask = mask.unsqueeze(-1)
        masked = emb * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
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
        )

        genre_first = self.pool_embedding(
            item_genre_ids, item_genre_mask, self.genre_first_order
        )

        dense_first = self.dense_linear(dense_features)

        return first_order_sum + genre_first + dense_first

    def fm_second_order(
        self,
        user_id, item_id, user_gender, user_age, user_occupation,
        item_genre_ids, item_genre_mask,
        history_item_ids, history_item_mask
    ):
        history_emb = self.pool_embedding(
            history_item_ids, history_item_mask, self.feature_embeddings["item_id"]
        )

        genre_emb = self.pool_embedding(
            item_genre_ids, item_genre_mask, self.genre_embedding
        )

        emb_list = [
            self.feature_embeddings["user_id"](user_id),
            self.feature_embeddings["item_id"](item_id),
            self.feature_embeddings["user_gender"](user_gender),
            self.feature_embeddings["user_age"](user_age),
            self.feature_embeddings["user_occupation"](user_occupation),
            genre_emb,
            history_emb,
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
        out = self.deep_output(hidden)
        return out

    def forward(
        self,
        user_id, item_id, user_gender, user_age, user_occupation,
        item_genre_ids, item_genre_mask,
        history_item_ids, history_item_mask,
        dense_features
    ):
        first_order = self.fm_first_order(
            user_id, item_id, user_gender, user_age, user_occupation,
            item_genre_ids, item_genre_mask,
            dense_features
        )

        second_order, sparse_emb_tensor = self.fm_second_order(
            user_id, item_id, user_gender, user_age, user_occupation,
            item_genre_ids, item_genre_mask,
            history_item_ids, history_item_mask
        )

        deep_out = self.deep_part(sparse_emb_tensor, dense_features)

        logits = first_order + second_order + deep_out
        return logits.squeeze(1)