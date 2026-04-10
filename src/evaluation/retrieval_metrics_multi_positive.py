import math
import torch
import pandas as pd
from tqdm import tqdm


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


@torch.no_grad()
def evaluate_two_tower_multi_positive(
    model,
    eval_df: pd.DataFrame,
    user_train_history: dict,
    k_list=(5, 10, 20),
    device="cuda"
):
    """
    多正样本召回评估

    指标：
    - HitRate@K: 是否至少命中 1 个正样本
    - Recall@K: 命中正样本数 / 总正样本数
    - NDCG@K: 命中的位置质量
    """

    model.eval()

    hit_count = {k: 0 for k in k_list}
    recall_sum = {k: 0.0 for k in k_list}
    ndcg_sum = {k: 0.0 for k in k_list}

    total_users = len(eval_df)

    for _, row in tqdm(eval_df.iterrows(), total=total_users, desc="评估多正样本召回"):
        user_id = int(row["user_id"])
        target_items = list(row["target_items"])
        target_set = set(target_items)
        candidate_items = list(row["candidate_items"])

        history_items = user_train_history[user_id]

        user_ids = torch.tensor(
            [user_id] * len(candidate_items),
            dtype=torch.long,
            device=device
        )

        item_ids = torch.tensor(
            candidate_items,
            dtype=torch.long,
            device=device
        )

        history_tensor = torch.tensor(history_items, dtype=torch.long, device=device)
        history_items_batch = history_tensor.unsqueeze(0).repeat(len(candidate_items), 1)
        history_mask = torch.ones_like(history_items_batch, dtype=torch.float)

        scores = model.score(
            user_ids=user_ids,
            history_item_ids=history_items_batch,
            history_mask=history_mask,
            item_ids=item_ids
        )

        sorted_indices = torch.argsort(scores, descending=True)
        ranked_items = [candidate_items[i] for i in sorted_indices.cpu().tolist()]
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