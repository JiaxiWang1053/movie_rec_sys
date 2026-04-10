import random
import pandas as pd
from src.data.negative_sampling import sample_negative_items


def build_retrieval_train_data_dynamic_hardneg(
    split_dict: dict,
    user_all_history: dict,
    all_movie_set: set,
    dynamic_hard_neg_pool: dict,
    random_seed: int = 42
):
    """
    构造动态 hard negative 训练数据。

    每个正样本生成两条训练样本：
    1. hard negative
    2. random negative

    返回 DataFrame 字段：
    - user_id
    - history_items
    - pos_item
    - neg_item
    """

    random.seed(random_seed)
    rows = []

    for user_id, split_info in split_dict.items():
        train_items = split_info["train"]

        for idx in range(1, len(train_items)):
            history_items = train_items[:idx]
            pos_item = train_items[idx]

            # ===== 动态 hard negative =====
            hard_candidates = dynamic_hard_neg_pool.get(user_id, [])
            hard_neg = None
            if len(hard_candidates) > 0:
                hard_neg = random.choice(hard_candidates)

            # ===== random negative =====
            random_neg_list = sample_negative_items(
                user_id=user_id,
                user_all_history=user_all_history,
                all_movie_set=all_movie_set,
                num_negatives=1,
                random_seed=random_seed + user_id + pos_item + idx
            )
            random_neg = random_neg_list[0]

            # 1. hard negative 样本
            if hard_neg is not None:
                rows.append({
                    "user_id": user_id,
                    "history_items": history_items,
                    "pos_item": pos_item,
                    "neg_item": hard_neg
                })

            # 2. random negative 样本
            rows.append({
                "user_id": user_id,
                "history_items": history_items,
                "pos_item": pos_item,
                "neg_item": random_neg
            })

    retrieval_train_df = pd.DataFrame(rows)
    return retrieval_train_df