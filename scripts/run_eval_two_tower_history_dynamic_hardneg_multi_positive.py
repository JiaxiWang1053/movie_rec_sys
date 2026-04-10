import torch

from src.models.two_tower_history import TwoTowerHistoryRetrievalModel
from src.data.load_movielens import load_ml_1m
from src.data.build_retrieval_eval_data_multi_positive import build_retrieval_eval_data_multi_positive
from src.evaluation.retrieval_metrics_multi_positive import evaluate_two_tower_multi_positive


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

    # 1. 读取基础信息
    _, users, movies = load_ml_1m(data_path)
    num_users = users["user_id"].max() + 1
    num_items = movies["movie_id"].max() + 1

    # 2. 构造模型
    model = TwoTowerHistoryRetrievalModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64
    ).to(device)

    model.load_state_dict(
        torch.load("checkpoints/two_tower_history_dynamic_hardneg.pth", map_location=device)
    )
    print("模型加载完成：two_tower_history_dynamic_hardneg.pth")

    # 3. 构造多正样本评估集
    eval_df, split_dict = build_retrieval_eval_data_multi_positive(
        data_path=data_path,
        positive_threshold=4,
        min_history=8,
        n_test=5,
        num_negatives=100,
        random_seed=42
    )

    user_train_history = {
        user_id: info["train"] for user_id, info in split_dict.items()
    }

    print("参与评估用户数：", len(eval_df))

    # 4. 评估
    metrics = evaluate_two_tower_multi_positive(
        model=model,
        eval_df=eval_df,
        user_train_history=user_train_history,
        k_list=(5, 10, 20),
        device=device
    )

    print("\n双塔 + History + Dynamic Hard Negative（多正样本评估）结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()