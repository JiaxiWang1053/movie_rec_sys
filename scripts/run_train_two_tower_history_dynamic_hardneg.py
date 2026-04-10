import torch
from src.training.train_two_tower_history_dynamic_hardneg import (
    train_two_tower_history_dynamic_hardneg
)


def main():
    data_path = "data/raw/ml-1m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("当前设备：", device)

    train_two_tower_history_dynamic_hardneg(
        data_path=data_path,
        embedding_dim=64,
        batch_size=512,
        learning_rate=1e-3,
        num_epochs=3,
        positive_threshold=4,
        min_history=3,
        hard_pool_size=50,
        candidate_sample_size=300,
        random_seed=42,
        device=device,
        save_path="checkpoints/two_tower_history_dynamic_hardneg.pth"
    )

    print("\n动态 Hard Negative 版本训练完成。")


if __name__ == "__main__":
    main()