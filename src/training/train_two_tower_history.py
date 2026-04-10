import torch
from torch.utils.data import DataLoader

from src.data.build_retrieval_train_data_with_history import build_retrieval_train_data_with_history
from src.data.retrieval_history_dataset import (
    RetrievalTrainHistoryDataset,
    collate_fn_with_history
)
from src.models.two_tower_history import TwoTowerHistoryRetrievalModel, bpr_loss
from src.data.load_movielens import load_ml_1m


def train_two_tower_history(
    data_path: str,
    embedding_dim: int = 64,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_epochs: int = 3,
    positive_threshold: int = 4,
    min_history: int = 3,
    num_negatives_per_positive: int = 1,
    random_seed: int = 42,
    device: str = "cuda"
):
    _, users, movies = load_ml_1m(data_path)
    num_users = users["user_id"].max() + 1
    num_items = movies["movie_id"].max() + 1

    retrieval_train_df = build_retrieval_train_data_with_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history,
        num_negatives_per_positive=num_negatives_per_positive,
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

    model = TwoTowerHistoryRetrievalModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    # ===== 保存模型 =====
    save_path = "checkpoints/two_tower_history.pth"

    import os
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(model.state_dict(), save_path)

    print(f"\n模型已保存到: {save_path}")

    return model