import pandas as pd
from src.data.load_movielens import load_ml_1m


def build_positive_samples(data_path: str, positive_threshold: int = 4):
    """
    构造推荐系统中的正样本数据。

    参数：
    - data_path: MovieLens 1M 数据目录，例如 "data/raw/ml-1m"
    - positive_threshold: 评分阈值，默认 rating >= 4 视为正样本

    返回：
    - positive_df: 只包含正样本的交互数据，字段为
        user_id, movie_id, timestamp
    - users: 用户表
    - movies: 电影表
    """

    # 1. 读取原始三张表
    ratings, users, movies = load_ml_1m(data_path)

    # 2. 只保留评分大于等于阈值的记录，作为正样本
    positive_df = ratings[ratings["rating"] >= positive_threshold].copy()

    # 3. 只保留推荐系统当前阶段需要的列
    positive_df = positive_df[["user_id", "movie_id", "timestamp"]]

    # 4. 按用户和时间排序
    #    这样做是为了后续构造用户行为序列，以及按时间切分训练/测试集
    positive_df = positive_df.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)

    return positive_df, users, movies