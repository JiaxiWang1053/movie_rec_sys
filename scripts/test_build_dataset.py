from data.build_positive_samples import build_positive_samples


def main():
    positive_df, users, movies = build_positive_samples("data/raw/ml-1m")

    print("正样本交互数据维度：", positive_df.shape)
    print("用户表维度：", users.shape)
    print("电影表维度：", movies.shape)

    print("\n正样本前5行：")
    print(positive_df.head())

    print("\n正样本中的用户数：", positive_df["user_id"].nunique())
    print("正样本中的电影数：", positive_df["movie_id"].nunique())
    print("正样本总数：", len(positive_df))


if __name__ == "__main__":
    main()