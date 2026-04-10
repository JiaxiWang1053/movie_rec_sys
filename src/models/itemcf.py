import math
from collections import defaultdict


class ItemCF:
    """
    基础版 Item-based Collaborative Filtering

    训练输入：
    - split_dict[user_id]["train"] 作为用户训练历史

    核心步骤：
    1. 统计每个 item 被多少用户喜欢
    2. 统计 item-item 共现次数
    3. 计算 item-item 相似度
    """

    def __init__(self):
        self.item_sim = defaultdict(dict)
        self.item_count = defaultdict(int)
        self.user_train_history = {}

    def fit(self, split_dict: dict):
        """
        基于训练历史构建 item-item 相似度表

        参数：
        - split_dict:
            {
                user_id: {
                    "train": [...],
                    "valid": ...,
                    "test": ...
                }
            }
        """
        self.user_train_history = {
            user_id: info["train"] for user_id, info in split_dict.items()
        }

        co_matrix = defaultdict(lambda: defaultdict(int))

        # 1. 统计 item 出现次数 和 item-item 共现次数
        for user_id, train_items in self.user_train_history.items():
            unique_items = list(set(train_items))

            for i in unique_items:
                self.item_count[i] += 1

            for i in unique_items:
                for j in unique_items:
                    if i == j:
                        continue
                    co_matrix[i][j] += 1

        # 2. 计算 item-item 相似度
        for i, related_items in co_matrix.items():
            for j, cij in related_items.items():
                self.item_sim[i][j] = cij / math.sqrt(
                    self.item_count[i] * self.item_count[j]
                )

    def score_candidates(self, user_id: int, candidate_items: list):
        """
        对某个用户的一组候选电影打分

        打分思路：
        - 看候选电影与用户训练历史中的电影有多相似
        - 相似度累加作为最终分数

        参数：
        - user_id
        - candidate_items: 候选电影列表

        返回：
        - scores: dict
            {item_id: score}
        """
        user_history = self.user_train_history.get(user_id, [])
        scores = {}

        for candidate in candidate_items:
            score = 0.0

            for hist_item in user_history:
                if hist_item in self.item_sim and candidate in self.item_sim[hist_item]:
                    score += self.item_sim[hist_item][candidate]

            scores[candidate] = score

        return scores