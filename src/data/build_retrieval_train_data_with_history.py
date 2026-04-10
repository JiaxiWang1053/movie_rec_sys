import pandas as pd
from src.data.train_valid_test_split import split_by_user_history
from src.data.negative_sampling import (
    build_negative_sampler_inputs,
    sample_negative_items
)


def build_retrieval_train_data_with_history(
    data_path: str,
    positive_threshold: int = 4,
    min_history: int = 3,
    num_negatives_per_positive: int = 1,
    random_seed: int = 42
):
    """
    构造带历史行为的召回训练数据。

    每条样本格式为：
    - user_id
    - history_items: 当前正样本之前的训练历史
    - pos_item
    - neg_item

    注意：
    - 只有当 history_items 非空时，才构造该训练样本
    """

    split_dict, _, _, _ = split_by_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history
    )

    user_all_history, all_movie_set = build_negative_sampler_inputs(data_path)

    rows = []

    for user_id, split_info in split_dict.items():
        train_items = split_info["train"]

        # 对 train 中的每个位置构造样本
        # 当前正样本之前的部分作为历史
        for idx in range(1, len(train_items)):
            history_items = train_items[:idx]
            pos_item = train_items[idx]

            negative_items = sample_negative_items(
                user_id=user_id,
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