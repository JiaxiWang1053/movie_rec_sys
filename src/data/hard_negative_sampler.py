from collections import defaultdict


def build_itemcf_hard_negative_pool(
    itemcf_model,
    user_train_history: dict,
    all_movie_set: set,
    topk: int = 200
):
    """
    为每个用户构建 hard negative 候选池（基于 ItemCF）

    返回：
    {
        user_id: [hard_negative_items]
    }
    """

    hard_neg_pool = {}

    for user_id, history_items in user_train_history.items():

        # 用 ItemCF 给用户推荐候选
        scores = defaultdict(float)

        for hist_item in history_items:
            if hist_item not in itemcf_model.item_sim:
                continue

            for related_item, sim in itemcf_model.item_sim[hist_item].items():
                scores[related_item] += sim

        # 排序取 TopK
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 过滤已看过的
        watched_set = set(history_items)

        hard_candidates = []
        for item, _ in ranked_items:
            if item not in watched_set:
                hard_candidates.append(item)

            if len(hard_candidates) >= topk:
                break

        hard_neg_pool[user_id] = hard_candidates

    return hard_neg_pool