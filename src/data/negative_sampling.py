import random
from src.data.load_movielens import load_ml_1m
from src.data.build_all_history import build_all_history


def build_all_movie_set(data_path: str):
    """
    构造电影全集。

    参数：
    - data_path: MovieLens 数据目录

    返回：
    - all_movie_set: set
        所有 movie_id 的集合
    """
    _, _, movies = load_ml_1m(data_path)
    all_movie_set = set(movies["movie_id"].unique())
    return all_movie_set


def sample_negative_items(
    user_id: int,
    user_all_history: dict,
    all_movie_set: set,
    num_negatives: int,
    random_seed: int = 42
):
    """
    为指定用户随机采样负样本电影。

    负样本定义：
    - 该电影属于电影全集
    - 该用户从未交互过（基于全部交互历史，而不是仅正样本历史）

    参数：
    - user_id: 用户ID
    - user_all_history: 用户全部交互历史字典
        形式为 {user_id: [movie_id1, movie_id2, ...]}
    - all_movie_set: 所有电影的集合
    - num_negatives: 要采样的负样本数量
    - random_seed: 随机种子，保证结果可复现

    返回：
    - negative_items: list
        采样得到的负样本 movie_id 列表
    """

    random.seed(random_seed)

    # 1. 取出该用户的全部已看电影
    watched_items = set(user_all_history.get(user_id, []))

    # 2. 构造该用户的负样本候选池：电影全集 - 已看集合
    negative_candidates = list(all_movie_set - watched_items)

    # 3. 如果候选池为空，直接返回空列表
    if len(negative_candidates) == 0:
        return []

    # 4. 如果候选池数量不足 num_negatives，就全部返回
    if len(negative_candidates) <= num_negatives:
        return negative_candidates

    # 5. 否则随机采样指定数量
    negative_items = random.sample(negative_candidates, num_negatives)

    return negative_items


def build_negative_sampler_inputs(data_path: str):
    """
    一次性构造负采样所需的基础输入：
    - user_all_history
    - all_movie_set

    参数：
    - data_path: MovieLens 数据目录

    返回：
    - user_all_history: dict
    - all_movie_set: set
    """
    user_all_history, _ = build_all_history(data_path)
    all_movie_set = build_all_movie_set(data_path)
    return user_all_history, all_movie_set