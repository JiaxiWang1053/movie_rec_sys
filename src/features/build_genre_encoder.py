from src.data.load_movielens import load_ml_1m


def build_genre_encoder(data_path: str):
    """
    构造 genre 编码器，并为每个电影建立 genre id 列表。

    返回：
    - genre2id: dict
        {genre_name: genre_id}
    - item_genre_ids_map: dict
        {item_id: [genre_id1, genre_id2, ...]}
    - num_genres: int
        genre 总数
    """

    _, _, movies = load_ml_1m(data_path)

    # 1. 收集所有 genre
    genre_set = set()
    for _, row in movies.iterrows():
        genres = str(row["genres"]).split("|")
        for g in genres:
            genre_set.add(g)

    genre_list = sorted(list(genre_set))
    genre2id = {genre: idx for idx, genre in enumerate(genre_list)}

    # 2. 建立 item -> [genre_id] 映射
    item_genre_ids_map = {}
    for _, row in movies.iterrows():
        item_id = int(row["movie_id"])
        genres = str(row["genres"]).split("|")
        genre_ids = [genre2id[g] for g in genres]
        item_genre_ids_map[item_id] = genre_ids

    num_genres = len(genre2id)

    return genre2id, item_genre_ids_map, num_genres