import math
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.training.train_deepfm_ranker_with_history import train_deepfm_ranker_with_history
from src.features.build_deepfm_ranking_dataset_with_history import build_deepfm_ranking_dataset_with_history
from src.data.deepfm_ranking_dataset_with_history import (
    DeepFMRankingHistoryDataset,
    collate_fn_deepfm_with_history
)
from src.models.deepfm_ranker_with_history import DeepFMRankerWithHistory


def dcg_at_k(labels, k):
    dcg = 0.0
    for i, rel in enumerate(labels[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(labels, k):
    ideal = sorted(labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(labels, k) / ideal_dcg


def apply_dense_stats(df, dense_feature_cols, dense_stats):
    df = df.copy()
    for col in dense_feature_cols:
        mean = dense_stats[col]["mean"]
        std = dense_stats[col]["std"]
        df[col] = (df[col] - mean) / std
    return df


def evaluate_deepfm(test_df, test_dataset, model, device="cuda", k_list=(5, 10, 20)):
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_deepfm_with_history
    )

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
                history_item_ids=batch["history_item_ids"].to(device),
                history_item_mask=batch["history_item_mask"].to(device),
                dense_features=batch["dense_features"].to(device),
            )

            preds = torch.sigmoid(logits).cpu().numpy()

            y_true_all.extend(labels.tolist())
            y_pred_all.extend(preds.tolist())

    auc = roc_auc_score(y_true_all, y_pred_all)

    eval_df = test_df.copy()
    eval_df["score"] = y_pred_all

    hit_count = {k: 0 for k in k_list}
    ndcg_sum = {k: 0.0 for k in k_list}
    total_users = eval_df["user_id"].nunique()

    for user_id, group in eval_df.groupby("user_id"):
        group = group.sort_values("score", ascending=False)
        labels = group["label"].tolist()

        for k in k_list:
            topk_labels = labels[:k]
            if sum(topk_labels) > 0:
                hit_count[k] += 1
            ndcg_sum[k] += ndcg_at_k(labels, k)

    metrics = {"AUC": auc}
    for k in k_list:
        metrics[f"HitRate@{k}"] = hit_count[k] / total_users
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / total_users

    return metrics


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

    train_deepfm_ranker_with_history(
        data_path=data_path,
        embedding_dim=16,
        hidden_dims=(128, 64),
        dropout=0.1,
        batch_size=512,
        learning_rate=1e-3,
        num_epochs=5,
        device=device,
        save_path="checkpoints/deepfm_ranker_with_history.pth"
    )

    test_df, sparse_feature_info, dense_feature_cols, item_genre_ids_map, num_genres = (
        build_deepfm_ranking_dataset_with_history(
            data_path=data_path,
            stage="test",
            num_negatives=99,
            retrieval_model_ckpt="checkpoints/two_tower_history_dynamic_hardneg.pth",
            device=device
        )
    )

    ckpt = torch.load("checkpoints/deepfm_ranker_with_history.pth", map_location=device)

    test_df = apply_dense_stats(test_df, dense_feature_cols, ckpt["dense_stats"])

    test_dataset = DeepFMRankingHistoryDataset(
        ranking_df=test_df,
        dense_feature_cols=dense_feature_cols,
        item_genre_ids_map=item_genre_ids_map
    )

    model = DeepFMRankerWithHistory(
        sparse_feature_info=ckpt["sparse_feature_info"],
        num_genres=ckpt["num_genres"],
        num_dense_features=len(ckpt["dense_feature_cols"]),
        embedding_dim=ckpt["embedding_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        dropout=ckpt["dropout"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("加载最佳 DeepFM+History 模型完成。")

    metrics = evaluate_deepfm(
        test_df=test_df,
        test_dataset=test_dataset,
        model=model,
        device=device,
        k_list=(5, 10, 20)
    )

    print("\nDeepFM + History 排序结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()