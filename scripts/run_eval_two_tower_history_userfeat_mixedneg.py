import torch

from src.models.two_tower_history_userfeat import TwoTowerHistoryUserFeatRetrievalModel
from src.data.load_movielens import load_ml_1m
from src.data.build_retrieval_eval_data import build_retrieval_eval_data
from src.data.train_valid_test_split import split_by_user_history
from src.evaluation.retrieval_metrics_history import evaluate_two_tower_with_history
from src.features.build_user_feature_tensors import build_user_feature_tensors


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

    # ===== 1. 读取基础信息 =====
    _, users, movies = load_ml_1m(data_path)
    num_users = users["user_id"].max() + 1
    num_items = movies["movie_id"].max() + 1

    # ===== 2. 构造用户特征张量 =====
    feature_tensors, feature_dims = build_user_feature_tensors(data_path)

    # ===== 3. 构造模型 =====
    model = TwoTowerHistoryUserFeatRetrievalModel(
        num_users=num_users,
        num_items=num_items,
        num_gender=feature_dims["num_gender"],
        num_age=feature_dims["num_age"],
        num_occupation=feature_dims["num_occupation"],
        gender_ids=feature_tensors["gender_ids"],
        age_ids=feature_tensors["age_ids"],
        occupation_ids=feature_tensors["occupation_ids"],
        embedding_dim=64,
        hidden_dim=128,
        user_feat_dim=16
    ).to(device)

    # ===== 4. 加载权重 =====
    model.load_state_dict(
        torch.load("checkpoints/two_tower_history_userfeat_mixedneg.pth", map_location=device)
    )
    print("模型加载完成：two_tower_history_userfeat_mixedneg.pth")

    # ===== 5. 获取 train history =====
    split_dict, _, _, _ = split_by_user_history(
        data_path=data_path,
        positive_threshold=4,
        min_history=3
    )

    user_train_history = {
        user_id: info["train"] for user_id, info in split_dict.items()
    }

    # ===== 6. 构造 valid 评估集 =====
    valid_eval_df = build_retrieval_eval_data(
        data_path=data_path,
        stage="valid",
        positive_threshold=4,
        min_history=3,
        num_negatives=99,
        random_seed=42
    )

    # ===== 7. 评估 =====
    metrics = evaluate_two_tower_with_history(
        model=model,
        eval_df=valid_eval_df,
        user_train_history=user_train_history,
        k_list=(5, 10, 20),
        device=device
    )

    print("\n双塔 + History + User Features + Mixed Negative 召回结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()