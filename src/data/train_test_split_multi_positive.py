from src.data.build_user_history import build_user_history


def split_by_user_history_multi_positive(
    data_path: str,
    positive_threshold: int = 4,
    min_history: int = 8,
    n_test: int = 5
):
    """
    多正样本评估切分

    对每个用户：
    - train = 前面的历史
    - test = 最后 n_test 个正样本

    返回：
    - split_dict:
        {
            user_id: {
                "train": [...],
                "test": [...]
            }
        }
    """

    user_positive_history, _ = build_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold
    )

    split_dict = {}

    for user_id, history in user_positive_history.items():
        if len(history) < min_history:
            continue

        train_items = history[:-n_test]
        test_items = history[-n_test:]

        if len(train_items) < 1:
            continue

        split_dict[user_id] = {
            "train": train_items,
            "test": test_items
        }

    return split_dict