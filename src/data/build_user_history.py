from src.data.build_positive_samples import build_positive_samples


def build_user_history(data_path: str, positive_threshold: int = 4):
    """
    根据正样本交互数据，构造每个用户的历史行为序列。

    参数：
    - data_path: MovieLens 1M 数据目录
    - positive_threshold: 正样本阈值，默认 rating >= 4

    返回：
    - user_history: dict
        键：user_id
        值：按时间排序后的 movie_id 列表
    - positive_df: 正样本交互表
    """

    # 1. 先构造正样本数据
    positive_df, _, _ = build_positive_samples(
        data_path=data_path,
        positive_threshold=positive_threshold
    )

    # 2. 按 user_id 聚合 movie_id，生成用户历史行为序列
    #    因为 positive_df 已经按 user_id 和 timestamp 排好序，
    #    所以这里聚合出来的列表天然就是时间有序的
    user_history_df = (
        positive_df.groupby("user_id")["movie_id"]
        .apply(list)
        .reset_index()
    )

    # 3. 转成字典形式，方便后续使用
    #    例如：
    #    user_history[1] = [3186, 1721, 1270, ...]
    user_history = dict(
        zip(user_history_df["user_id"], user_history_df["movie_id"])
    )

    return user_history, positive_df