import os
import lightgbm as lgb


def train_lightgbm_ranker(
    train_df,
    feature_cols,
    group_train,
    save_path="checkpoints/lightgbm_ranker.txt"
):
    """
    训练 LightGBM 排序模型（LGBMRanker）
    """

    X_train = train_df[feature_cols]
    y_train = train_df["label"]

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        group=group_train
    )

    os.makedirs("checkpoints", exist_ok=True)
    model.booster_.save_model(save_path)
    print(f"LightGBM ranker 已保存到: {save_path}")

    return model