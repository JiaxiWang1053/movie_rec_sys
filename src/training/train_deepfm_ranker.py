import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.features.build_deepfm_ranking_dataset import build_deepfm_ranking_dataset
from src.data.deepfm_ranking_dataset import DeepFMRankingDataset, collate_fn_deepfm
from src.models.deepfm_ranker import DeepFMRanker


def standardize_dense_features(train_df, test_df, dense_feature_cols):
    """
    用 train 统计量对 dense 特征做标准化
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    stats = {}
    for col in dense_feature_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()

        if std < 1e-6:
            std = 1.0

        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std

        stats[col] = {"mean": float(mean), "std": float(std)}

    return train_df, test_df, stats


def train_deepfm_ranker(
    data_path: str,
    embedding_dim: int = 16,
    hidden_dims=(128, 64),
    dropout: float = 0.1,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_epochs: int = 5,
    device: str = "cuda",
    save_path: str = "checkpoints/deepfm_ranker.pth"
):
    """
    训练 DeepFM 排序模型（稳定版）
    """

    # ===== 1. 构造训练集(valid) =====
    train_df, sparse_feature_info, dense_feature_cols, item_genre_ids_map, num_genres = (
        build_deepfm_ranking_dataset(
            data_path=data_path,
            stage="valid",
            num_negatives=99,
            retrieval_model_ckpt="checkpoints/two_tower_history_dynamic_hardneg.pth",
            device=device
        )
    )

    # ===== 2. 构造测试集(test)，用于监控 =====
    test_df, _, _, _, _ = build_deepfm_ranking_dataset(
        data_path=data_path,
        stage="test",
        num_negatives=99,
        retrieval_model_ckpt="checkpoints/two_tower_history_dynamic_hardneg.pth",
        device=device
    )

    # ===== 3. dense 特征标准化 =====
    train_df, test_df, dense_stats = standardize_dense_features(
        train_df, test_df, dense_feature_cols
    )

    train_dataset = DeepFMRankingDataset(
        ranking_df=train_df,
        dense_feature_cols=dense_feature_cols,
        item_genre_ids_map=item_genre_ids_map
    )
    test_dataset = DeepFMRankingDataset(
        ranking_df=test_df,
        dense_feature_cols=dense_feature_cols,
        item_genre_ids_map=item_genre_ids_map
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_deepfm
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_deepfm
    )

    # ===== 4. 模型 =====
    model = DeepFMRanker(
        sparse_feature_info=sparse_feature_info,
        num_genres=num_genres,
        num_dense_features=len(dense_feature_cols),
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_auc = -1.0

    # ===== 5. 训练 =====
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            labels = batch["label"].to(device)

            logits = model(
                user_id=batch["user_id"].to(device),
                item_id=batch["item_id"].to(device),
                user_gender=batch["user_gender"].to(device),
                user_age=batch["user_age"].to(device),
                user_occupation=batch["user_occupation"].to(device),
                item_genre_ids=batch["item_genre_ids"].to(device),
                item_genre_mask=batch["item_genre_mask"].to(device),
                dense_features=batch["dense_features"].to(device),
            )

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== 6. 每轮评估 AUC =====
        model.eval()
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for batch in test_loader:
                labels = batch["label"].cpu().numpy()

                logits = model(
                    user_id=batch["user_id"].to(device),
                    item_id=batch["item_id"].to(device),
                    user_gender=batch["user_gender"].to(device),
                    user_age=batch["user_age"].to(device),
                    user_occupation=batch["user_occupation"].to(device),
                    item_genre_ids=batch["item_genre_ids"].to(device),
                    item_genre_mask=batch["item_genre_mask"].to(device),
                    dense_features=batch["dense_features"].to(device),
                )

                preds = torch.sigmoid(logits).cpu().numpy()

                y_true_all.extend(labels.tolist())
                y_pred_all.extend(preds.tolist())

        auc = roc_auc_score(y_true_all, y_pred_all)
        print(f"Epoch [{epoch+1}/{num_epochs}] - train loss: {avg_loss:.6f} - test AUC: {auc:.6f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dense_feature_cols": dense_feature_cols,
                    "dense_stats": dense_stats,
                    "sparse_feature_info": sparse_feature_info,
                    "num_genres": num_genres,
                    "embedding_dim": embedding_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                },
                save_path
            )
            print(f"保存当前最佳 DeepFM 模型到: {save_path}")

    return model