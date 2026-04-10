import torch
from torch.utils.data import Dataset


class DeepFMRankingHistoryDataset(Dataset):
    def __init__(
        self,
        ranking_df,
        dense_feature_cols,
        item_genre_ids_map
    ):
        self.labels = ranking_df["label"].values.astype("float32")

        self.user_ids = ranking_df["user_id"].values
        self.item_ids = ranking_df["item_id"].values
        self.user_gender = ranking_df["user_gender"].values
        self.user_age = ranking_df["user_age"].values
        self.user_occupation = ranking_df["user_occupation"].values

        self.dense_features = ranking_df[dense_feature_cols].values.astype("float32")

        self.history_items = ranking_df["history_items"].values
        self.item_genre_ids_map = item_genre_ids_map

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item_id = int(self.item_ids[idx])
        genre_ids = self.item_genre_ids_map.get(item_id, [])
        history_ids = list(self.history_items[idx])

        return {
            "label": torch.tensor(self.labels[idx], dtype=torch.float),

            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item_id": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "user_gender": torch.tensor(self.user_gender[idx], dtype=torch.long),
            "user_age": torch.tensor(self.user_age[idx], dtype=torch.long),
            "user_occupation": torch.tensor(self.user_occupation[idx], dtype=torch.long),

            "dense_features": torch.tensor(self.dense_features[idx], dtype=torch.float),

            "item_genre_ids": torch.tensor(genre_ids, dtype=torch.long),
            "history_item_ids": torch.tensor(history_ids, dtype=torch.long),
        }


def collate_fn_deepfm_with_history(batch):
    labels = torch.stack([x["label"] for x in batch], dim=0)

    user_id = torch.stack([x["user_id"] for x in batch], dim=0)
    item_id = torch.stack([x["item_id"] for x in batch], dim=0)
    user_gender = torch.stack([x["user_gender"] for x in batch], dim=0)
    user_age = torch.stack([x["user_age"] for x in batch], dim=0)
    user_occupation = torch.stack([x["user_occupation"] for x in batch], dim=0)

    dense_features = torch.stack([x["dense_features"] for x in batch], dim=0)

    # genres padding
    genre_lengths = [len(x["item_genre_ids"]) for x in batch]
    max_genre_len = max(genre_lengths) if len(genre_lengths) > 0 else 1

    padded_genres = []
    genre_masks = []

    for x in batch:
        g = x["item_genre_ids"]
        pad_len = max_genre_len - len(g)

        padded = torch.cat([g, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        mask = torch.cat([torch.ones(len(g), dtype=torch.float),
                          torch.zeros(pad_len, dtype=torch.float)], dim=0)

        padded_genres.append(padded)
        genre_masks.append(mask)

    padded_genres = torch.stack(padded_genres, dim=0)
    genre_masks = torch.stack(genre_masks, dim=0)

    # history padding
    hist_lengths = [len(x["history_item_ids"]) for x in batch]
    max_hist_len = max(hist_lengths) if len(hist_lengths) > 0 else 1

    padded_histories = []
    history_masks = []

    for x in batch:
        h = x["history_item_ids"]
        pad_len = max_hist_len - len(h)

        padded = torch.cat([h, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        mask = torch.cat([torch.ones(len(h), dtype=torch.float),
                          torch.zeros(pad_len, dtype=torch.float)], dim=0)

        padded_histories.append(padded)
        history_masks.append(mask)

    padded_histories = torch.stack(padded_histories, dim=0)
    history_masks = torch.stack(history_masks, dim=0)

    return {
        "label": labels,

        "user_id": user_id,
        "item_id": item_id,
        "user_gender": user_gender,
        "user_age": user_age,
        "user_occupation": user_occupation,

        "dense_features": dense_features,

        "item_genre_ids": padded_genres,
        "item_genre_mask": genre_masks,

        "history_item_ids": padded_histories,
        "history_item_mask": history_masks,
    }