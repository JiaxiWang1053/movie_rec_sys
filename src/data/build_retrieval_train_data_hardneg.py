import random
import pandas as pd

from src.data.train_valid_test_split import split_by_user_history
from src.data.negative_sampling import build_negative_sampler_inputs, sample_negative_items
from src.models.itemcf import ItemCF
from src.data.hard_negative_sampler import build_itemcf_hard_negative_pool


def sample_mixed_negative(
    user_id: int,
    hard_neg_pool: dict,
    user_all_history: dict,
    all_movie_set: set,
    num_negatives: int = 1,
    random_seed: int = 42
):
    """
    混合负采样：
    1. 优先从 hard negative pool 里采样
    2. 如果不够，再用随机负样本补足
    """
    random.seed(random_seed)

    hard_candidates = hard_neg_pool.get(user_id, [])

    sampled_negatives = []

    # 1. 先从 hard negative 里采
    if len(hard_candidates) > 0:
        if len(hard_candidates) <= num_negatives:
            sampled_negatives.extend(hard_candidates)
        else:
            sampled_negatives.extend(random.sample(hard_candidates, num_negatives))

    # 2. 不够的话，再用随机负样本补
    remain = num_negatives - len(sampled_negatives)
    if remain > 0:
        random_negatives = sample_negative_items(
            user_id=user_id,
            user_all_history=user_all_history,
            all_movie_set=all_movie_set,
            num_negatives=remain,
            random_seed=random_seed + 999
        )

        # 避免重复
        existing = set(sampled_negatives)
        for item in random_negatives:
            if item not in existing:
                sampled_negatives.append(item)
                existing.add(item)
            if len(sampled_negatives) >= num_negatives:
                break

    return sampled_negatives


def build_retrieval_train_data_hardneg(
    data_path: str,
    positive_threshold: int = 4,
    min_history: int = 3,
    hard_negative_topk: int = 200,
    num_negatives_per_positive: int = 1,
    random_seed: int = 42
):
    """
    构造带 hard negative 的召回训练数据。

    每条样本格式：
    - user_id
    - history_items
    - pos_item
    - neg_item
    """

    # 1. 切分数据
    split_dict, _, _, _ = split_by_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history
    )

    user_train_history = {
        user_id: info["train"] for user_id, info in split_dict.items()
    }

    # 2. 构造 ItemCF
    itemcf = ItemCF()
    itemcf.fit(split_dict)

    # 3. 负采样基础输入
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

        for idx in range(1, len(train_items)):
            history_items = train_items[:idx]
            pos_item = train_items[idx]

            negative_items = sample_mixed_negative(
                user_id=user_id,
                hard_neg_pool=hard_neg_pool,
                user_all_history=user_all_history,
                all_movie_set=all_movie_set,
                num_negatives=num_negatives_per_positive,
                random_seed=random_seed + user_id + pos_item + idx
            )

            for neg_item in negative_items:
                rows.append({
                    "user_id": user_id,
                    "history_items": history_items,
                    "pos_item": pos_item,
                    "neg_item": neg_item
                })

    retrieval_train_df = pd.DataFrame(rows)
    return retrieval_train_df