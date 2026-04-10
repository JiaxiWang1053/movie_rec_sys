import pandas as pd
from src.data.train_valid_test_split import split_by_user_history
from src.data.negative_sampling import (
    build_negative_sampler_inputs,
    sample_negative_items
)


def build_retrieval_eval_data(
    data_path: str,
    stage: str = "valid",
    positive_threshold: int = 4,
    min_history: int = 3,
    num_negatives: int = 99,
    random_seed: int = 42
):
    """
    构造召回阶段的评估数据。

    每个用户构造一条评估样本：
    - 1 个目标正样本（valid 或 test）
    - num_negatives 个随机负样本
    - candidate_items = [target_item] + negative_items

    参数：
    - data_path: MovieLens 数据目录
    - stage: "valid" 或 "test"
    - positive_threshold: 正样本阈值
    - min_history: 用户最小正样本历史长度
    - num_negatives: 每个用户评估时采样的负样本数量
    - random_seed: 随机种子

    返回：
    - eval_df: DataFrame
        字段包括：
        user_id, target_item, negative_items, candidate_items
    """

    assert stage in ["valid", "test"], "stage 只能是 'valid' 或 'test'"

    # 1. 获取时间切分结果
    split_dict, train_df, valid_df, test_df = split_by_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history
    )

    # 2. 获取负采样输入
    user_all_history, all_movie_set = build_negative_sampler_inputs(data_path)

    rows = []

    # 3. 逐用户构造评估候选集
    for user_id, split_info in split_dict.items():
        target_item = split_info[stage]

        negative_items = sample_negative_items(
            user_id=user_id,
            user_all_history=user_all_history,
            all_movie_set=all_movie_set,
            num_negatives=num_negatives,
            random_seed=random_seed + user_id
        )

        # 候选集 = 1个正样本 + 多个负样本
        candidate_items = [target_item] + negative_items

        rows.append({
            "user_id": user_id,
            "target_item": target_item,
            "negative_items": negative_items,
            "candidate_items": candidate_items
        })

    eval_df = pd.DataFrame(rows)

    return eval_df