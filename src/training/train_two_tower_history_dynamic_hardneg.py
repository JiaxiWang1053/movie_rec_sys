import os
import torch
from torch.utils.data import DataLoader

from src.data.train_valid_test_split import split_by_user_history
from src.data.negative_sampling import build_negative_sampler_inputs
from src.data.build_retrieval_train_data_dynamic_hardneg import (
    build_retrieval_train_data_dynamic_hardneg
)
from src.data.retrieval_history_dataset import (
    RetrievalTrainHistoryDataset,
    collate_fn_with_history
)
from src.data.dynamic_hard_negative import build_dynamic_hard_negative_pool
from src.models.two_tower_history import (
    TwoTowerHistoryRetrievalModel,
    bpr_loss
)
from src.data.load_movielens import load_ml_1m


def train_two_tower_history_dynamic_hardneg(
    data_path: str,
    embedding_dim: int = 64,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_epochs: int = 3,
    positive_threshold: int = 4,
    min_history: int = 3,
    hard_pool_size: int = 50,
    candidate_sample_size: int = 300,
    random_seed: int = 42,
    device: str = "cuda",
    save_path: str = "checkpoints/two_tower_history_dynamic_hardneg.pth"
):
    """
    TwoTower + History + Dynamic Hard Negative

    训练流程：
    每个 epoch 开始时：
    1. 用当前模型动态构造 hard negative pool
    2. 基于该 pool + random negative 构造训练数据
    3. 训练 1 个 epoch
    """

    # 1. 基础信息
    _, users, movies = load_ml_1m(data_path)
    num_users = users["user_id"].max() + 1
    num_items = movies["movie_id"].max() + 1
    all_movie_list = movies["movie_id"].unique().tolist()

    # 2. 切分数据
    split_dict, _, _, _ = split_by_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history
    )

    user_train_history = {
        user_id: info["train"] for user_id, info in split_dict.items()
    }

    # 3. 全量历史与电影全集
    user_all_history, all_movie_set = build_negative_sampler_inputs(data_path)

    # 4. 模型
    model = TwoTowerHistoryRetrievalModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 逐 epoch 动态更新 hard negatives
    for epoch in range(num_epochs):
        print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")

        # ===== 5.1 构造动态 hard negative pool =====
        dynamic_hard_neg_pool = build_dynamic_hard_negative_pool(
            model=model,
            user_train_history=user_train_history,
            user_all_history=user_all_history,
            all_movie_list=all_movie_list,
            hard_pool_size=hard_pool_size,
            candidate_sample_size=candidate_sample_size,
            device=device,
            random_seed=random_seed + epoch
        )

        # ===== 5.2 构造训练数据 =====
        retrieval_train_df = build_retrieval_train_data_dynamic_hardneg(
            split_dict=split_dict,
            user_all_history=user_all_history,
            all_movie_set=all_movie_set,
            dynamic_hard_neg_pool=dynamic_hard_neg_pool,
            random_seed=random_seed + epoch
        )

        dataset = RetrievalTrainHistoryDataset(retrieval_train_df)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_with_history
        )

        # ===== 5.3 训练一个 epoch =====
        model.train()
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

    # 6. 保存模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存到: {save_path}")

    return model