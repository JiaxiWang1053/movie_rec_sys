import torch
import pandas as pd
from tqdm import tqdm


@torch.no_grad()
def evaluate_two_tower_retrieval(
    model,
    eval_df: pd.DataFrame,
    k_list=(5, 10, 20),
    device="cuda"
):
    """
    评估双塔召回模型。

    参数：
    - model: 已训练好的双塔模型
    - eval_df: 召回评估数据，包含
        user_id, target_item, negative_items, candidate_items
    - k_list: 需要评估的 K 值列表，例如 (5, 10, 20)
    - device: 运行设备

    返回：
    - metrics: dict
        例如：
        {
            "HitRate@5": 0.12,
            "HitRate@10": 0.21,
            "HitRate@20": 0.35,
            "Recall@5": 0.12,
            ...
        }
    """

    model.eval()

    hit_count = {k: 0 for k in k_list}
    total_users = len(eval_df)

    for _, row in tqdm(eval_df.iterrows(), total=total_users, desc="评估召回"):
        user_id = int(row["user_id"])
        target_item = int(row["target_item"])
        candidate_items = row["candidate_items"]

        # 1. 构造当前用户和候选物品的张量
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

        # 2. 计算候选集打分
        scores = model.score(user_ids, item_ids)  # [num_candidates]

        # 3. 按分数从高到低排序
        sorted_indices = torch.argsort(scores, descending=True)
        ranked_items = [candidate_items[i] for i in sorted_indices.cpu().tolist()]

        # 4. 计算各个 K 下是否命中
        for k in k_list:
            topk_items = ranked_items[:k]
            if target_item in topk_items:
                hit_count[k] += 1

    # 5. 汇总指标
    metrics = {}
    for k in k_list:
        hitrate = hit_count[k] / total_users
        recall = hit_count[k] / total_users   # 当前每用户只有1个正样本，因此 recall = hitrate

        metrics[f"HitRate@{k}"] = hitrate
        metrics[f"Recall@{k}"] = recall

    return metrics