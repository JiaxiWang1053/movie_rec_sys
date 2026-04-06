from src.data.load_movielens import load_ml_1m
from src.data.build_positive_samples import build_positive_samples
from src.data.build_user_history import build_user_history
from src.data.train_valid_test_split import split_by_user_history
from src.data.build_all_history import build_all_history


def main():
    data_path = "data/raw/ml-1m"

    print("=" * 60)
    print("1. 测试原始数据读取")
    print("=" * 60)
    ratings, users, movies = load_ml_1m(data_path)

    print("ratings 维度：", ratings.shape)
    print("users 维度：", users.shape)
    print("movies 维度：", movies.shape)

    assert len(ratings) > 0, "ratings 为空"
    assert len(users) > 0, "users 为空"
    assert len(movies) > 0, "movies 为空"

    print("\n原始数据读取成功。\n")

    print("=" * 60)
    print("2. 测试正样本构造")
    print("=" * 60)
    positive_df, users2, movies2 = build_positive_samples(
        data_path=data_path,
        positive_threshold=4
    )

    print("正样本交互维度：", positive_df.shape)
    print("正样本中的用户数：", positive_df['user_id'].nunique())
    print("正样本中的电影数：", positive_df['movie_id'].nunique())

    assert len(positive_df) > 0, "正样本为空"
    assert list(positive_df.columns) == ["user_id", "movie_id", "timestamp"], "正样本字段不正确"

    print("\n正样本构造成功。\n")

    print("=" * 60)
    print("3. 测试正样本用户历史构造")
    print("=" * 60)
    user_positive_history, positive_df_2 = build_user_history(
        data_path=data_path,
        positive_threshold=4
    )

    print("正样本历史用户数：", len(user_positive_history))

    sample_users = list(user_positive_history.keys())[:3]
    for user_id in sample_users:
        history = user_positive_history[user_id]
        print(f"user_id={user_id}, 正样本历史长度={len(history)}, 前10个={history[:10]}")
        assert len(history) > 0, f"user {user_id} 的正样本历史为空"

    print("\n正样本用户历史构造成功。\n")

    print("=" * 60)
    print("4. 测试按时间切分 train / valid / test")
    print("=" * 60)
    split_dict, train_df, valid_df, test_df = split_by_user_history(
        data_path=data_path,
        positive_threshold=4,
        min_history=3
    )

    print("参与切分的用户数：", len(split_dict))
    print("train_df 行数：", len(train_df))
    print("valid_df 行数：", len(valid_df))
    print("test_df 行数：", len(test_df))

    assert len(split_dict) > 0, "切分结果为空"
    assert len(valid_df) == len(split_dict), "valid_df 行数应等于用户数"
    assert len(test_df) == len(split_dict), "test_df 行数应等于用户数"

    for user_id in list(split_dict.keys())[:3]:
        info = split_dict[user_id]
        print(f"\nuser_id={user_id}")
        print("train:", info["train"][:10], "..." if len(info["train"]) > 10 else "")
        print("valid:", info["valid"])
        print("test :", info["test"])

        assert len(info["train"]) >= 1, f"user {user_id} 的 train 历史为空"

    print("\n时间切分成功。\n")

    print("=" * 60)
    print("5. 测试全部交互历史构造")
    print("=" * 60)
    user_all_history, all_interactions_df = build_all_history(data_path)

    print("完整交互数：", len(all_interactions_df))
    print("全部历史中的用户数：", len(user_all_history))

    assert len(all_interactions_df) == len(ratings), "完整交互数应与 ratings 行数一致"
    assert len(user_all_history) == ratings["user_id"].nunique(), "全部历史用户数不一致"

    for user_id in list(user_all_history.keys())[:3]:
        history = user_all_history[user_id]
        print(f"user_id={user_id}, 全部历史长度={len(history)}, 前10个={history[:10]}")
        assert len(history) > 0, f"user {user_id} 的全部历史为空"

    print("\n全部交互历史构造成功。\n")

    print("=" * 60)
    print("6. 做一个关键一致性检查")
    print("=" * 60)
    # 检查：正样本历史长度一定不大于全部历史长度
    checked_users = 0
    for user_id in user_positive_history:
        if user_id in user_all_history:
            pos_len = len(user_positive_history[user_id])
            all_len = len(user_all_history[user_id])
            assert pos_len <= all_len, f"user {user_id} 的正样本历史竟然长于全部历史"
            checked_users += 1

    print(f"已检查 {checked_users} 个用户：正样本历史长度 <= 全部历史长度")
    print("\n一致性检查成功。\n")

    print("=" * 60)
    print("所有基础数据模块测试通过。")
    print("=" * 60)


if __name__ == "__main__":
    main()