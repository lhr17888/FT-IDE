import gensim.downloader as api
import numpy as np
wv = api.load('word2vec-google-news-300')

vec_king = wv['TaylorLautner']
vec_queen = wv['phone']
print(np.dot(vec_king, vec_queen) / (np.linalg.norm(vec_king) * np.linalg.norm(vec_queen)))