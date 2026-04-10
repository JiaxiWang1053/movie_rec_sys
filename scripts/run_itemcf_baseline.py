from src.data.train_valid_test_split import split_by_user_history
from src.data.build_retrieval_eval_data import build_retrieval_eval_data
from src.models.itemcf import ItemCF


def evaluate_itemcf(itemcf_model, eval_df, k_list=(5, 10, 20)):
    """
    评估 ItemCF baseline
    """
    hit_count = {k: 0 for k in k_list}
    total_users = len(eval_df)

    for _, row in eval_df.iterrows():
        user_id = int(row["user_id"])
        target_item = int(row["target_item"])
        candidate_items = row["candidate_items"]

        scores = itemcf_model.score_candidates(user_id, candidate_items)

        ranked_items = sorted(
            candidate_items,
            key=lambda x: scores.get(x, 0.0),
            reverse=True
        )

        for k in k_list:
            topk_items = ranked_items[:k]
            if target_item in topk_items:
                hit_count[k] += 1

    metrics = {}
    for k in k_list:
        hitrate = hit_count[k] / total_users
        recall = hitrate  # 当前每用户只有1个目标正样本
        metrics[f"HitRate@{k}"] = hitrate
        metrics[f"Recall@{k}"] = recall

    return metrics


def main():
    data_path = "data/raw/ml-1m"

    # 1. 构造训练切分
    split_dict, train_df, valid_df, test_df = split_by_user_history(
        data_path=data_path,
        positive_threshold=4,
        min_history=3
    )

    # 2. 训练 ItemCF
    itemcf = ItemCF()
    itemcf.fit(split_dict)

    # 3. 构造 valid 评估集
    valid_eval_df = build_retrieval_eval_data(
        data_path=data_path,
        stage="valid",
        positive_threshold=4,
        min_history=3,
        num_negatives=99,
        random_seed=42
    )

    # 4. 评估
    metrics = evaluate_itemcf(
        itemcf_model=itemcf,
        eval_df=valid_eval_df,
        k_list=(5, 10, 20)
    )

    print("ItemCF baseline 召回评估结果：")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")


if __name__ == "__main__":
    main()