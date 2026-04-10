import re
import math
import torch
import pandas as pd

from src.data.load_movielens import load_ml_1m
from src.data.train_valid_test_split import split_by_user_history
from src.data.build_retrieval_eval_data import build_retrieval_eval_data
from src.models.two_tower_history import TwoTowerHistoryRetrievalModel


def extract_year_from_title(title: str):
    """
    从电影标题中提取年份，例如 'Toy Story (1995)' -> 1995
    """
    match = re.search(r"\((\d{4})\)", str(title))
    if match:
        return int(match.group(1))
    return 0


def build_item_popularity(split_dict: dict):
    """
    基于 train history 统计 item popularity
    """
    item_popularity = {}

    for _, info in split_dict.items():
        for item in info["train"]:
            item_popularity[item] = item_popularity.get(item, 0) + 1

    return item_popularity


def build_user_genre_history(split_dict: dict, movie_genres_map: dict):
    """
    为每个用户构造历史 genre 集合（使用 train history）
    """
    user_genre_history = {}

    for user_id, info in split_dict.items():
        genre_set = set()
        for item in info["train"]:
            genre_set.update(movie_genres_map.get(item, set()))
        user_genre_history[user_id] = genre_set

    return user_genre_history


@torch.no_grad()
def compute_retrieval_scores_for_candidates(
    model,
    user_id: int,
    history_items: list,
    candidate_items: list,
    device: str = "cuda"
):
    """
    用训练好的召回模型给候选集打分
    """
    model.eval()

    user_ids = torch.tensor(
        [user_id] * len(candidate_items),
        dtype=torch.long,
        device=device
    )

    item_ids = torch.tensor(
        candidate_items,
        dtype=torch.long,
        device=device
    )

    history_tensor = torch.tensor(history_items, dtype=torch.long, device=device)
    history_items_batch = history_tensor.unsqueeze(0).repeat(len(candidate_items), 1)
    history_mask = torch.ones_like(history_items_batch, dtype=torch.float)

    scores = model.score(
        user_ids=user_ids,
        history_item_ids=history_items_batch,
        history_mask=history_mask,
        item_ids=item_ids
    )

    return scores.detach().cpu().tolist()


def build_ranking_dataset(
    data_path: str,
    stage: str = "valid",
    num_negatives: int = 99,
    positive_threshold: int = 4,
    min_history: int = 3,
    retrieval_model_ckpt: str = "checkpoints/two_tower_history_dynamic_hardneg.pth",
    device: str = "cuda"
):
    """
    构造 LightGBM 排序数据集。

    思路：
    - 对每个用户，使用 candidate_items（1正样本 + 99负样本）
    - 展开为逐行样本：(user, item, features, label)

    返回：
    - ranking_df
    - feature_cols
    - group_sizes
    """

    assert stage in ["valid", "test"]

    # ===== 1. 读取基础数据 =====
    ratings, users, movies = load_ml_1m(data_path)

    # ===== 2. 切分数据，拿 train history =====
    split_dict, _, _, _ = split_by_user_history(
        data_path=data_path,
        positive_threshold=positive_threshold,
        min_history=min_history
    )

    user_train_history = {
        user_id: info["train"] for user_id, info in split_dict.items()
    }

    # ===== 3. 构造候选集 =====
    eval_df = build_retrieval_eval_data(
        data_path=data_path,
        stage=stage,
        positive_threshold=positive_threshold,
        min_history=min_history,
        num_negatives=num_negatives,
        random_seed=42
    )

    # ===== 4. 加载最佳召回模型，用其分数作为排序特征 =====
    num_users = users["user_id"].max() + 1
    num_items = movies["movie_id"].max() + 1

    retrieval_model = TwoTowerHistoryRetrievalModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64
    ).to(device)

    retrieval_model.load_state_dict(
        torch.load(retrieval_model_ckpt, map_location=device)
    )
    retrieval_model.eval()

    # ===== 5. 构造用户/物品侧辅助特征 =====
    users_df = users.copy()
    users_df["gender_code"] = users_df["gender"].map({"M": 0, "F": 1})

    user_feature_map = {}
    for _, row in users_df.iterrows():
        user_feature_map[int(row["user_id"])] = {
            "user_gender": int(row["gender_code"]),
            "user_age": int(row["age"]),
            "user_occupation": int(row["occupation"]),
        }

    movies_df = movies.copy()
    movies_df["item_year"] = movies_df["title"].apply(extract_year_from_title)

    movie_genres_map = {}
    movie_year_map = {}

    for _, row in movies_df.iterrows():
        item_id = int(row["movie_id"])
        genres = set(str(row["genres"]).split("|")) if pd.notna(row["genres"]) else set()
        movie_genres_map[item_id] = genres
        movie_year_map[item_id] = int(row["item_year"])

    item_popularity = build_item_popularity(split_dict)
    user_genre_history = build_user_genre_history(split_dict, movie_genres_map)

    # ===== 6. 展开候选集，构造排序样本 =====
    rows = []

    for _, row in eval_df.iterrows():
        user_id = int(row["user_id"])
        target_item = int(row["target_item"])
        candidate_items = row["candidate_items"]

        history_items = user_train_history[user_id]
        retrieval_scores = compute_retrieval_scores_for_candidates(
            model=retrieval_model,
            user_id=user_id,
            history_items=history_items,
            candidate_items=candidate_items,
            device=device
        )

        user_feat = user_feature_map[user_id]
        user_history_len = len(history_items)
        user_genres = user_genre_history[user_id]

        for item_id, retrieval_score in zip(candidate_items, retrieval_scores):
            item_id = int(item_id)

            item_pop = item_popularity.get(item_id, 0)
            item_year = movie_year_map.get(item_id, 0)
            item_genres = movie_genres_map.get(item_id, set())
            genre_overlap_count = len(user_genres.intersection(item_genres))

            label = 1 if item_id == target_item else 0

            rows.append({
                "user_id": user_id,
                "item_id": item_id,
                "label": label,

                # 排序特征
                "retrieval_score": float(retrieval_score),
                "user_history_len": int(user_history_len),
                "item_popularity": int(item_pop),
                "genre_overlap_count": int(genre_overlap_count),
                "user_gender": int(user_feat["user_gender"]),
                "user_age": int(user_feat["user_age"]),
                "user_occupation": int(user_feat["user_occupation"]),
                "item_year": int(item_year),
            })

    ranking_df = pd.DataFrame(rows)

    feature_cols = [
        "retrieval_score",
        "user_history_len",
        "item_popularity",
        "genre_overlap_count",
        "user_gender",
        "user_age",
        "user_occupation",
        "item_year",
    ]

    # 每个用户一个 group（用于 ranker）
    group_sizes = ranking_df.groupby("user_id").size().tolist()

    return ranking_df, feature_cols, group_sizes