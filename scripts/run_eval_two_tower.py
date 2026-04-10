import torch

from src.training.train_two_tower import train_two_tower
from src.data.build_retrieval_eval_data import build_retrieval_eval_data
from src.evaluation.retrieval_metrics import evaluate_two_tower_retrieval


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

    # 1. 先训练一个最基础的双塔模型
    model = train_two_tower(
        data_path=data_path,
        embedding_dim=64,
        batch_size=1024,
        learning_rate=1e-3,
        num_epochs=3,
        positive_threshold=4,
        min_history=3,
        num_negatives_per_positive=1,
        random_seed=42,
        device=device
    )

    # 2. 构造 valid 评估集
    valid_eval_df = build_retrieval_eval_data(
        data_path=data_path,
        stage="valid",
        positive_threshold=4,
        min_history=3,
        num_negatives=99,
        random_seed=42
    )

    # 3. 评估
    metrics = evaluate_two_tower_retrieval(
        model=model,
        eval_df=valid_eval_df,
        k_list=(5, 10, 20),
        device=device
    )

    print("\n召回评估结果：")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")


if __name__ == "__main__":
    main()