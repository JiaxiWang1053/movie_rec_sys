import torch

from src.models.two_tower_history import TwoTowerHistoryRetrievalModel
from src.data.load_movielens import load_ml_1m
from src.data.build_retrieval_eval_data import build_retrieval_eval_data
from src.data.train_valid_test_split import split_by_user_history
from src.evaluation.retrieval_metrics_history import evaluate_two_tower_with_history


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

    # 1. 构造模型
    _, users, movies = load_ml_1m(data_path)
    num_users = users["user_id"].max() + 1
    num_items = movies["movie_id"].max() + 1

    model = TwoTowerHistoryRetrievalModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64
    ).to(device)

    # 2. 加载 hard negative 版本权重
    model.load_state_dict(
        torch.load("checkpoints/two_tower_history_hardneg.pth", map_location=device)
    )
    print("模型加载完成：two_tower_history_hardneg.pth")

    # 3. 获取 train history
    split_dict, _, _, _ = split_by_user_history(
        data_path=data_path,
        positive_threshold=4,
        min_history=3
    )

    user_train_history = {
        user_id: info["train"] for user_id, info in split_dict.items()
    }

    # 4. 构造 valid 评估集
    valid_eval_df = build_retrieval_eval_data(
        data_path=data_path,
        stage="valid",
        positive_threshold=4,
        min_history=3,
        num_negatives=99,
        random_seed=42
    )

    # 5. 评估
    metrics = evaluate_two_tower_with_history(
        model=model,
        eval_df=valid_eval_df,
        user_train_history=user_train_history,
        k_list=(5, 10, 20),
        device=device
    )

    print("\n双塔 + History + Hard Negative 召回结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()