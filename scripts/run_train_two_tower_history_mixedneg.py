import torch
from src.training.train_two_tower_history_mixedneg import train_two_tower_history_mixedneg


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

    train_two_tower_history_mixedneg(
        data_path=data_path,
        embedding_dim=64,
        batch_size=512,
        learning_rate=1e-3,
        num_epochs=3,
        positive_threshold=4,
        min_history=3,
        hard_negative_topk=200,
        random_seed=42,
        device=device,
        save_path="checkpoints/two_tower_history_mixedneg.pth"
    )

    print("\nMixed Negative 增强版双塔训练完成。")


if __name__ == "__main__":
    main()