import torch
import pandas as pd
from tqdm import tqdm


@torch.no_grad()
def evaluate_two_tower_with_history(
    model,
    eval_df: pd.DataFrame,
    user_train_history: dict,
    k_list=(5, 10, 20),
    device="cuda"
):
    """
    带用户历史的双塔召回评估
    """

    model.eval()

    hit_count = {k: 0 for k in k_list}
    total_users = len(eval_df)

    for _, row in tqdm(eval_df.iterrows(), total=total_users, desc="评估召回(History)"):

        user_id = int(row["user_id"])
        target_item = int(row["target_item"])
        candidate_items = row["candidate_items"]

        history_items = user_train_history[user_id]

        # ===== 构造张量 =====
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

        # 扩展 history 到 batch
        history_items_batch = history_tensor.unsqueeze(0).repeat(len(candidate_items), 1)

        history_mask = torch.ones_like(history_items_batch, dtype=torch.float)

        # ===== 打分 =====
        scores = model.score(
            user_ids=user_ids,
            history_item_ids=history_items_batch,
            history_mask=history_mask,
            item_ids=item_ids
        )

        # 排序
        sorted_indices = torch.argsort(scores, descending=True)
        ranked_items = [candidate_items[i] for i in sorted_indices.cpu().tolist()]

        # 计算指标
        for k in k_list:
            if target_item in ranked_items[:k]:
                hit_count[k] += 1

    # 汇总
    metrics = {}
    for k in k_list:
        hitrate = hit_count[k] / total_users
        metrics[f"HitRate@{k}"] = hitrate
        metrics[f"Recall@{k}"] = hitrate

    return metrics