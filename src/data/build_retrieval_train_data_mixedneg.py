import random
import pandas as pd

from src.data.train_valid_test_split import split_by_user_history
from src.data.negative_sampling import build_negative_sampler_inputs, sample_negative_items
from src.models.itemcf import ItemCF
from src.data.hard_negative_sampler import build_itemcf_hard_negative_pool


def build_retrieval_train_data_mixedneg(
    data_path: str,
    positive_threshold: int = 4,
    min_history: int = 3,
    hard_negative_topk: int = 200,
    random_seed: int = 42
):
    """
    Mixed Negative Sampling 版本的召回训练数据构造。

    每个正样本会生成两条训练样本：
    1. (user, history, pos, hard_neg)
    2. (user, history, pos, random_neg)

    返回 DataFrame 字段：
    - user_id
    - history_items
    - pos_item
    - neg_item
    """

    random.seed(random_seed)

    # 1. 切分数据
    split_dict, _, _, _ = split_by_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history
    )

    user_train_history = {
        user_id: info["train"] for user_id, info in split_dict.items()
    }

    # 2. 训练 ItemCF，用来生成 hard negative pool
    itemcf = ItemCF()
    itemcf.fit(split_dict)

    # 3. 获取负采样基础输入
    user_all_history, all_movie_set = build_negative_sampler_inputs(data_path)

    # 4. 构造 hard negative pool
    hard_neg_pool = build_itemcf_hard_negative_pool(
        itemcf_model=itemcf,
        user_train_history=user_train_history,
        all_movie_set=all_movie_set,
        topk=hard_negative_topk
    )

    rows = []

    # 5. 构造训练样本
    for user_id, split_info in split_dict.items():
        train_items = split_info["train"]

        # history 版训练：当前正样本之前必须有历史
        for idx in range(1, len(train_items)):
            history_items = train_items[:idx]
            pos_item = train_items[idx]

            # ===== hard negative =====
            hard_candidates = hard_neg_pool.get(user_id, [])

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

            # 1) hard negative 样本
            if hard_neg is not None:
                rows.append({
                    "user_id": user_id,
                    "history_items": history_items,
                    "pos_item": pos_item,
                    "neg_item": hard_neg
                })

            # 2) random negative 样本
            rows.append({
                "user_id": user_id,
                "history_items": history_items,
                "pos_item": pos_item,
                "neg_item": random_neg
            })

    retrieval_train_df = pd.DataFrame(rows)
    return retrieval_train_df