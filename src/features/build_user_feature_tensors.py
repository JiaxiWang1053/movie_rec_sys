import torch
from src.data.load_movielens import load_ml_1m


def build_user_feature_tensors(data_path: str):
    """
    根据 users.dat 构造按 user_id 索引的用户特征张量。

    返回：
    - feature_tensors: dict，包含
        - gender_ids: [num_users]
        - age_ids: [num_users]
        - occupation_ids: [num_users]
    - feature_dims: dict，包含各特征 embedding 表大小
    """

    _, users, _ = load_ml_1m(data_path)

    num_users = users["user_id"].max() + 1

    # ===== 1. gender 编码 =====
    gender_map = {"M": 0, "F": 1}

    # ===== 2. age 编码（压缩成连续 id）=====
    unique_ages = sorted(users["age"].unique().tolist())
    age_map = {age_value: idx for idx, age_value in enumerate(unique_ages)}

    # ===== 3. occupation 编码（压缩成连续 id）=====
    unique_occupations = sorted(users["occupation"].unique().tolist())
    occupation_map = {
        occ_value: idx for idx, occ_value in enumerate(unique_occupations)
    }

    # ===== 4. 初始化按 user_id 索引的特征张量 =====
    gender_ids = torch.zeros(num_users, dtype=torch.long)
    age_ids = torch.zeros(num_users, dtype=torch.long)
    occupation_ids = torch.zeros(num_users, dtype=torch.long)

    # user_id=0 在 MovieLens 中通常不用，这里保留默认值即可
    for _, row in users.iterrows():
        user_id = int(row["user_id"])

        gender_ids[user_id] = gender_map[row["gender"]]
        age_ids[user_id] = age_map[row["age"]]
        occupation_ids[user_id] = occupation_map[row["occupation"]]

    feature_tensors = {
        "gender_ids": gender_ids,
        "age_ids": age_ids,
        "occupation_ids": occupation_ids,
    }

    feature_dims = {
        "num_gender": len(gender_map),
        "num_age": len(age_map),
        "num_occupation": len(occupation_map),
    }

    return feature_tensors, feature_dims