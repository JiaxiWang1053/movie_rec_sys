from src.data.build_user_history import build_user_history
import pandas as pd


def split_by_user_history(
    data_path: str,
    positive_threshold: int = 4,
    min_history: int = 3
):
    """
    按用户历史行为长度进行过滤，并按时间切分 train / valid / test。

    参数：
    - data_path: MovieLens 数据目录
    - positive_threshold: 正样本阈值，默认 rating >= 4
    - min_history: 用户最少正样本数，默认至少为 3

    返回：
    - split_dict:
        字典，格式为
        {
            user_id: {
                "train": [movie_id1, movie_id2, ...],
                "valid": movie_id,
                "test": movie_id
            }
        }
    - train_df:
        训练交互表，字段为 user_id, movie_id
    - valid_df:
        验证交互表，字段为 user_id, movie_id
    - test_df:
        测试交互表，字段为 user_id, movie_id
    """

    # 1. 先构造用户历史序列
    user_history, _ = build_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold
    )

    split_dict = {}

    train_rows = []
    valid_rows = []
    test_rows = []

    # 2. 遍历每个用户的历史行为
    for user_id, history in user_history.items():
        # 只保留历史长度足够的用户
        if len(history) < min_history:
            continue

        # 训练集：前 n-2 条
        train_items = history[:-2]

        # 验证集：倒数第 2 条
        valid_item = history[-2]

        # 测试集：倒数第 1 条
        test_item = history[-1]

        # 保存到字典中，方便后续查用户历史
        split_dict[user_id] = {
            "train": train_items,
            "valid": valid_item,
            "test": test_item
        }

        # 展开训练集，形成行式数据
        for movie_id in train_items:
            train_rows.append({
                "user_id": user_id,
                "movie_id": movie_id
            })

        # 验证集和测试集各保存一条
        valid_rows.append({
            "user_id": user_id,
            "movie_id": valid_item
        })

        test_rows.append({
            "user_id": user_id,
            "movie_id": test_item
        })

    # 3. 转成 DataFrame，方便后续训练和评估
    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)
    test_df = pd.DataFrame(test_rows)

    return split_dict, train_df, valid_df, test_df