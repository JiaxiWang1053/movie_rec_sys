import torch
from torch.utils.data import Dataset


class RetrievalTrainHistoryDataset(Dataset):
    """
    带历史行为的双塔召回训练数据集。
    """

    def __init__(self, retrieval_train_df):
        self.user_ids = retrieval_train_df["user_id"].values
        self.history_items = retrieval_train_df["history_items"].values
        self.pos_item_ids = retrieval_train_df["pos_item"].values
        self.neg_item_ids = retrieval_train_df["neg_item"].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "history_items": torch.tensor(self.history_items[idx], dtype=torch.long),
            "pos_item_id": torch.tensor(self.pos_item_ids[idx], dtype=torch.long),
            "neg_item_id": torch.tensor(self.neg_item_ids[idx], dtype=torch.long),
        }


def collate_fn_with_history(batch):
    """
    处理变长 history_items 的 collate_fn
    """
    user_ids = torch.stack([x["user_id"] for x in batch], dim=0)
    pos_item_ids = torch.stack([x["pos_item_id"] for x in batch], dim=0)
    neg_item_ids = torch.stack([x["neg_item_id"] for x in batch], dim=0)

    history_lengths = [len(x["history_items"]) for x in batch]
    max_len = max(history_lengths)

    padded_histories = []
    history_masks = []

    for x in batch:
        hist = x["history_items"]
        pad_len = max_len - len(hist)

        padded = torch.cat([
            hist,
            torch.zeros(pad_len, dtype=torch.long)
        ], dim=0)

        mask = torch.cat([
            torch.ones(len(hist), dtype=torch.float),
            torch.zeros(pad_len, dtype=torch.float)
        ], dim=0)

        padded_histories.append(padded)
        history_masks.append(mask)

    padded_histories = torch.stack(padded_histories, dim=0)   # [B, L]
    history_masks = torch.stack(history_masks, dim=0)         # [B, L]

    return {
        "user_id": user_ids,
        "history_items": padded_histories,
        "history_mask": history_masks,
        "pos_item_id": pos_item_ids,
        "neg_item_id": neg_item_ids,
    }