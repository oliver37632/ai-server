import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('j_df.csv', encoding='utf-8')

df.kakao_blog_review_txt = df.kakao_blog_review_txt.fillna('없음')

count_vect_category = CountVectorizer(min_df=0, ngram_range=(1, 2))
place_category = count_vect_category.fit_transform(df['cate_mix'])
place_simi_cate = cosine_similarity(place_category, place_category)
place_simi_cate_sorted_ind = place_simi_cate.argsort()[:, ::-1]

count_vect_review = CountVectorizer(min_df=2, ngram_range=(1, 2))
place_review = count_vect_review.fit_transform(df['kakao_blog_review_txt'])

place_simi_review = cosine_similarity(place_review, place_review)
place_simi_review_sorted_ind = place_simi_review.argsort()[:, ::-1]
place_simi_co = (
    + place_simi_cate * 0.3
    + place_simi_review * 1
    + np.repeat([df['kakao_blog_review_qty'].values],
                len(df['kakao_blog_review_qty']), axis=0) * 0.001
    + np.repeat([df['kakao_star_point'].values],
                len(df['kakao_star_point']), axis=0) * 0.0001
    + np.repeat([df['kakao_star_point_qty'].values],
                len(df['kakao_star_point_qty']), axis=0) * 0.001
)

count_vect_category = CountVectorizer(min_df=0, ngram_range=(1, 2))
place_category = count_vect_category.fit_transform(df['cate_mix'])
place_simi_cate = cosine_similarity(place_category, place_category)
place_simi_cate_sorted_ind = place_simi_cate.argsort()[:, ::-1]

place_simi_co_sorted_ind = place_simi_co.argsort()[:, ::-1]


def find_simi_place(df, sorted_ind, place_name, top_n=10):

    place_title = df[df['name'] == place_name]
    place_index = place_title.index.values
    similar_indexes = sorted_ind[place_index, :(top_n)]
    similar_indexes = similar_indexes.reshape(-1)
    return df.iloc[similar_indexes]


app = Flask(__name__)


@app.route('/')
def man():
    return {
        "message": "tlqkf"
    }


@app.route('/RecomSys', methods=['POST'])
def home():
    fc_name = request.args.get('franc_name')
    f_name = fc_name

    sim_name = find_simi_place(df, place_simi_co_sorted_ind, f_name, 1)
    return str(sim_name.kakao_store_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
