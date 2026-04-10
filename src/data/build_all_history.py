from src.data.load_movielens import load_ml_1m


def build_all_history(data_path: str):
    """
    基于完整评分数据构造每个用户的全部交互历史。

    这里的“全部交互”指的是：
    - 只要用户对电影打过分
    - 不管评分高低
    - 都算这个用户“看过”该电影

    参数：
    - data_path: MovieLens 1M 数据目录，例如 "data/raw/ml-1m"

    返回：
    - user_all_history: dict
        键：user_id
        值：按时间排序的 movie_id 列表
    - all_interactions_df: DataFrame
        完整交互表，仅保留 user_id, movie_id, timestamp
    """

    # 1. 读取原始数据
    ratings, users, movies = load_ml_1m(data_path)

    # 2. 只保留构造交互历史需要的列
    all_interactions_df = ratings[["user_id", "movie_id", "timestamp"]].copy()

    # 3. 按用户和时间排序
    #    这样后面聚合成列表时，顺序就是从早到晚
    all_interactions_df = all_interactions_df.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)

    # 4. 按 user_id 聚合电影列表
    user_all_history_df = (
        all_interactions_df.groupby("user_id")["movie_id"]
        .apply(list)
        .reset_index()
    )

    # 5. 转成字典，方便后续负样本采样时快速查询
    user_all_history = dict(
        zip(user_all_history_df["user_id"], user_all_history_df["movie_id"])
    )

    return user_all_history, all_interactions_df