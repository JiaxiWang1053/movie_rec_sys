import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.features.build_ranking_dataset import build_ranking_dataset
from src.training.train_lightgbm_ranker import train_lightgbm_ranker


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


def evaluate_lightgbm_ranker(test_df, feature_cols, model, k_list=(5, 10, 20)):
    """
    评估 LightGBM 排序模型
    """
    X_test = test_df[feature_cols]
    y_true = test_df["label"].values
    y_score = model.predict(X_test)

    # 全局 AUC
    auc = roc_auc_score(y_true, y_score)

    # 逐用户排序评估
    eval_df = test_df.copy()
    eval_df["score"] = y_score

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

    metrics = {
        "AUC": auc
    }

    for k in k_list:
        metrics[f"HitRate@{k}"] = hit_count[k] / total_users
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / total_users

    return metrics


def main():
    data_path = "data/raw/ml-1m"

    # ===== 1. 构造训练集（valid）=====
    print("构造 LightGBM 排序训练集（valid）...")
    train_df, feature_cols, group_train = build_ranking_dataset(
        data_path=data_path,
        stage="valid",
        num_negatives=99,
        retrieval_model_ckpt="checkpoints/two_tower_history_dynamic_hardneg.pth",
        device="cuda"
    )

    print("训练集大小：", len(train_df))
    print("特征列：", feature_cols)

    # ===== 2. 训练 Ranker =====
    model = train_lightgbm_ranker(
        train_df=train_df,
        feature_cols=feature_cols,
        group_train=group_train,
        save_path="checkpoints/lightgbm_ranker.txt"
    )

    print("\n特征重要性（Feature Importance）：")

    import pandas as pd

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print(importance_df)

    # ===== 3. 构造测试集（test）=====
    print("\n构造 LightGBM 排序测试集（test）...")
    test_df, _, _ = build_ranking_dataset(
        data_path=data_path,
        stage="test",
        num_negatives=99,
        retrieval_model_ckpt="checkpoints/two_tower_history_dynamic_hardneg.pth",
        device="cuda"
    )

    print("测试集大小：", len(test_df))

    # ===== 4. 评估 =====
    metrics = evaluate_lightgbm_ranker(
        test_df=test_df,
        feature_cols=feature_cols,
        model=model,
        k_list=(5, 10, 20)
    )

    print("\nLightGBM 排序结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()