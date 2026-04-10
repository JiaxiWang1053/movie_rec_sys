import os
import torch
from torch.utils.data import DataLoader

from src.data.build_retrieval_train_data_mixedneg import build_retrieval_train_data_mixedneg
from src.data.retrieval_history_dataset import (
    RetrievalTrainHistoryDataset,
    collate_fn_with_history
)
from src.models.two_tower_history_userfeat import (
    TwoTowerHistoryUserFeatRetrievalModel,
    bpr_loss
)
from src.data.load_movielens import load_ml_1m
from src.features.build_user_feature_tensors import build_user_feature_tensors


def train_two_tower_history_userfeat_mixedneg(
    data_path: str,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    user_feat_dim: int = 16,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_epochs: int = 3,
    positive_threshold: int = 4,
    min_history: int = 3,
    hard_negative_topk: int = 200,
    random_seed: int = 42,
    device: str = "cuda",
    save_path: str = "checkpoints/two_tower_history_userfeat_mixedneg.pth"
):
    """
    训练：
    Two-Tower + History + User Features + Mixed Negative
    """

    # ===== 1. 读取原始数据 =====
    _, users, movies = load_ml_1m(data_path)
    num_users = users["user_id"].max() + 1
    num_items = movies["movie_id"].max() + 1

    # ===== 2. 构造用户特征张量 =====
    feature_tensors, feature_dims = build_user_feature_tensors(data_path)

    # ===== 3. 构造 mixed negative 训练数据 =====
    retrieval_train_df = build_retrieval_train_data_mixedneg(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history,
        hard_negative_topk=hard_negative_topk,
        random_seed=random_seed
    )

    dataset = RetrievalTrainHistoryDataset(retrieval_train_df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_with_history
    )

    # ===== 4. 模型 =====
    model = TwoTowerHistoryUserFeatRetrievalModel(
        num_users=num_users,
        num_items=num_items,
        num_gender=feature_dims["num_gender"],
        num_age=feature_dims["num_age"],
        num_occupation=feature_dims["num_occupation"],
        gender_ids=feature_tensors["gender_ids"],
        age_ids=feature_tensors["age_ids"],
        occupation_ids=feature_tensors["occupation_ids"],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        user_feat_dim=user_feat_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ===== 5. 训练 =====
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            user_ids = batch["user_id"].to(device)
            history_items = batch["history_items"].to(device)
            history_mask = batch["history_mask"].to(device)
            pos_item_ids = batch["pos_item_id"].to(device)
            neg_item_ids = batch["neg_item_id"].to(device)

            optimizer.zero_grad()

            pos_scores, neg_scores = model(
                user_ids=user_ids,
                history_item_ids=history_items,
                history_mask=history_mask,
                pos_item_ids=pos_item_ids,
                neg_item_ids=neg_item_ids
            )

            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - 平均训练损失: {avg_loss:.6f}")

    # ===== 6. 保存模型 =====
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存到: {save_path}")

    return model