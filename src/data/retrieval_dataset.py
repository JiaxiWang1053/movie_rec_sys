import torch
from torch.utils.data import Dataset


class RetrievalTrainDataset(Dataset):
    """
    双塔召回训练数据集。

    输入的 DataFrame 需要包含三列：
    - user_id
    - pos_item
    - neg_item
    """

    def __init__(self, retrieval_train_df):
        self.user_ids = retrieval_train_df["user_id"].values
        self.pos_item_ids = retrieval_train_df["pos_item"].values
        self.neg_item_ids = retrieval_train_df["neg_item"].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "pos_item_id": torch.tensor(self.pos_item_ids[idx], dtype=torch.long),
            "neg_item_id": torch.tensor(self.neg_item_ids[idx], dtype=torch.long),
        }