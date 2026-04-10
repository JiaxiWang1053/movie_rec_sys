import math

from src.models.itemcf import ItemCF
from src.data.build_retrieval_eval_data_multi_positive import build_retrieval_eval_data_multi_positive


def dcg_at_k(binary_relevance, k):
    dcg = 0.0
    for i, rel in enumerate(binary_relevance[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(binary_relevance, num_relevant, k):
    ideal = [1] * min(num_relevant, k)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(binary_relevance, k) / ideal_dcg


def train_itemcf_from_multi_positive_split(split_dict):
    """
    用 multi-positive split 的 train 部分训练 ItemCF
    """
    itemcf = ItemCF()

    # ItemCF.fit 需要：
    # {
    #   user_id: {
    #       "train": [...],
    #       ...
    #   }
    # }
    itemcf.fit(split_dict)
    return itemcf


def evaluate_itemcf_multi_positive(itemcf_model, eval_df, k_list=(5, 10, 20)):
    """
    多正样本评估 ItemCF

    指标：
    - HitRate@K: 至少命中1个正样本
    - Recall@K: 命中正样本数 / 总正样本数
    - NDCG@K: 命中的位置质量
    """
    hit_count = {k: 0 for k in k_list}
    recall_sum = {k: 0.0 for k in k_list}
    ndcg_sum = {k: 0.0 for k in k_list}

    total_users = len(eval_df)

    for _, row in eval_df.iterrows():
        user_id = int(row["user_id"])
        target_items = list(row["target_items"])
        target_set = set(target_items)
        candidate_items = list(row["candidate_items"])

        scores = itemcf_model.score_candidates(user_id, candidate_items)

        ranked_items = sorted(
            candidate_items,
            key=lambda x: scores.get(x, 0.0),
            reverse=True
        )

        ranked_binary = [1 if item in target_set else 0 for item in ranked_items]
        num_relevant = len(target_items)

        for k in k_list:
            topk_binary = ranked_binary[:k]
            hit_num = sum(topk_binary)

            if hit_num > 0:
                hit_count[k] += 1

            recall_sum[k] += hit_num / num_relevant
            ndcg_sum[k] += ndcg_at_k(ranked_binary, num_relevant, k)

    metrics = {}
    for k in k_list:
        metrics[f"HitRate@{k}"] = hit_count[k] / total_users
        metrics[f"Recall@{k}"] = recall_sum[k] / total_users
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / total_users

    return metrics


def main():
    data_path = "data/raw/ml-1m"

    # 1. 构造 multi-positive 评估集与 split
    eval_df, split_dict = build_retrieval_eval_data_multi_positive(
        data_path=data_path,
        positive_threshold=4,
        min_history=8,
        n_test=5,
        num_negatives=100,
        random_seed=42
    )

    print("参与评估用户数：", len(eval_df))

    # 2. 训练 ItemCF
    itemcf = train_itemcf_from_multi_positive_split(split_dict)

    # 3. 评估
    metrics = evaluate_itemcf_multi_positive(
        itemcf_model=itemcf,
        eval_df=eval_df,
        k_list=(5, 10, 20)
    )

    print("\nItemCF（多正样本评估）结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
