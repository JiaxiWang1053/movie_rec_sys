import pandas as pd
from src.data.train_valid_test_split import split_by_user_history
from src.data.negative_sampling import (
    build_negative_sampler_inputs,
    sample_negative_items
)


def build_retrieval_train_data(
    data_path: str,
    positive_threshold: int = 4,
    min_history: int = 3,
    num_negatives_per_positive: int = 1,
    random_seed: int = 42
):
    """
    构造召回阶段的训练数据，形式为：
    (user_id, pos_item, neg_item)

    参数：
    - data_path: MovieLens 数据目录
    - positive_threshold: 正样本阈值
    - min_history: 至少保留多少条正样本历史的用户
    - num_negatives_per_positive: 每个正样本配多少个负样本
    - random_seed: 随机种子

    返回：
    - retrieval_train_df: DataFrame
        字段为：
        user_id, pos_item, neg_item
    """

    # 1. 获取按时间切分后的用户训练历史
    split_dict, train_df, valid_df, test_df = split_by_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history
    )

    # 2. 获取负采样所需输入
    user_all_history, all_movie_set = build_negative_sampler_inputs(data_path)

    rows = []

    # 3. 遍历每个用户
    for user_id, split_info in split_dict.items():
        train_items = split_info["train"]

        # 4. 对该用户训练历史中的每个正样本，采样若干负样本
        for pos_item in train_items:
            negative_items = sample_negative_items(
                user_id=user_id,
                user_all_history=user_all_history,
                all_movie_set=all_movie_set,
                num_negatives=num_negatives_per_positive,
                random_seed=random_seed + user_id + pos_item
            )

            for neg_item in negative_items:
                rows.append({
                    "user_id": user_id,
                    "pos_item": pos_item,
                    "neg_item": neg_item
                })

    retrieval_train_df = pd.DataFrame(rows)

    return retrieval_train_df