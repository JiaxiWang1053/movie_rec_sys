import pandas as pd
from src.data.negative_sampling import build_negative_sampler_inputs, sample_negative_items
from src.data.train_test_split_multi_positive import split_by_user_history_multi_positive


def build_retrieval_eval_data_multi_positive(
    data_path: str,
    positive_threshold: int = 4,
    min_history: int = 8,
    n_test: int = 5,
    num_negatives: int = 100,
    random_seed: int = 42
):
    """
    构造多正样本召回评估数据

    每个用户：
    - target_items = 多个 test 正样本
    - candidate_items = target_items + negative_items

    返回：
    - eval_df:
        user_id, target_items, negative_items, candidate_items
    - split_dict
    """

    split_dict = split_by_user_history_multi_positive(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history,
        n_test=n_test
    )

    user_all_history, all_movie_set = build_negative_sampler_inputs(data_path)

    rows = []

    for user_id, split_info in split_dict.items():
        target_items = split_info["test"]

        negative_items = sample_negative_items(
            user_id=user_id,
            user_all_history=user_all_history,
            all_movie_set=all_movie_set,
            num_negatives=num_negatives,
            random_seed=random_seed + user_id
        )

        candidate_items = list(target_items) + list(negative_items)

        rows.append({
            "user_id": user_id,
            "target_items": target_items,
            "negative_items": negative_items,
            "candidate_items": candidate_items
        })

    eval_df = pd.DataFrame(rows)
    return eval_df, split_dict