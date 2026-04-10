import torch
from src.training.train_two_tower import train_two_tower


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

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

    print("\n双塔训练完成。")


if __name__ == "__main__":
    main()