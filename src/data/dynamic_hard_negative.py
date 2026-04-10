import random
import torch
from tqdm import tqdm


@torch.no_grad()
def build_dynamic_hard_negative_pool(
    model,
    user_train_history: dict,
    user_all_history: dict,
    all_movie_list: list,
    hard_pool_size: int = 50,
    candidate_sample_size: int = 300,
    device: str = "cuda",
    random_seed: int = 42
):
    """
    使用当前模型动态构造 hard negative pool。

    对每个用户：
    1. 从未交互电影中随机抽一批 candidate_sample_size 个候选
    2. 用当前模型打分
    3. 取分数最高的 hard_pool_size 个作为 dynamic hard negatives

    返回：
    {
        user_id: [hard_neg_item1, hard_neg_item2, ...]
    }
    """

    random.seed(random_seed)
    model.eval()

    hard_neg_pool = {}

    for user_id, train_history in tqdm(user_train_history.items(), desc="构造动态 hard negative pool"):
        watched_set = set(user_all_history.get(user_id, []))
        unobserved_items = [item for item in all_movie_list if item not in watched_set]

        if len(unobserved_items) == 0:
            hard_neg_pool[user_id] = []
            continue

        # 1. 先随机抽一批候选，避免全量打分太慢
        if len(unobserved_items) > candidate_sample_size:
            candidate_items = random.sample(unobserved_items, candidate_sample_size)
        else:
            candidate_items = unobserved_items

        # 2. 构造模型输入
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

        history_tensor = torch.tensor(train_history, dtype=torch.long, device=device)
        history_items_batch = history_tensor.unsqueeze(0).repeat(len(candidate_items), 1)
        history_mask = torch.ones_like(history_items_batch, dtype=torch.float)

        # 3. 当前模型打分
        scores = model.score(
            user_ids=user_ids,
            history_item_ids=history_items_batch,
            history_mask=history_mask,
            item_ids=item_ids
        )

        # 4. 选 top 分数最高的若干个作为 hard negatives
        sorted_indices = torch.argsort(scores, descending=True)
        top_indices = sorted_indices[:hard_pool_size].cpu().tolist()

        hard_items = [candidate_items[i] for i in top_indices]
        hard_neg_pool[user_id] = hard_items

    return hard_neg_pool