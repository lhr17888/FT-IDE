import numpy as np
import torch
import ollama
import gensim
from gensim.models import KeyedVectors,word2vec,Word2Vec


''' -------------------------- 提取实体名 -------------------------- '''

# count = 0
# with open('./data/ENTITY/4/entity2id.txt') as entityfile, open('./data/ENTITY/4/tttt.txt', 'w') as newfile:
#
#     for i in entityfile.readlines():
#         if count < 12414:
#             count += 1
#             continue
#         count += 1
#         print(count)
#         elements = i.split('\t')[0]
#         name = ''
#         flag = False
#         with open('./data/mid2name.txt', encoding='utf-8') as namefile:
#             for j in namefile.readlines():
#                 if elements == j.split('\t')[0]:
#                     name = j.split('\t')[1]
#                     flag = True
#                     print(name)
#                     break
#         if not flag:
#             newfile.write(f"{elements}\tNone\n")
#             continue
#         newfile.write(f"{elements}\t{name}")


''' -------------------------- 以上 -------------------------- '''


''' -------------------------- 提取实体向量（word2vec） -------------------------- '''

# import gensim.downloader as api
# import numpy as np
# import ollama
#
# wv = api.load('word2vec-google-news-300')
#
# with open('./data/ENTITY/4/tttt.txt', 'r') as ttttfile, \
#         open('./data/ENTITY/4/vectors.txt', 'w') as vectorfile:
#     for i in ttttfile.readlines():
#         name = i.strip().split('\t')[1]
#         if name.strip() not in wv:
#             vectorfile.write('\n')
#             continue
#         if name.strip() == 'None':
#             vectorfile.write('\n')
#             continue
#         vec = wv[name.strip()]
#         vec_str = '\t'.join(str(x) for x in vec)  # 将向量元素连接成字符串
#         vectorfile.write(f"{vec_str}\n")


# with open('./data/ENTITY/4/vectors.txt','r') as vectorfile:
#     eee_str = vectorfile.readlines()[7].strip().split('\t')
#     eee = np.array([float(x) for x in eee_str])
#     print(eee)
#     fff = 1
#     print(np.dot(eee, fff) / (np.linalg.norm(eee) * np.linalg.norm(fff)))

''' -------------------------- 以上 -------------------------- '''


''' -------------------------- 提取实体向量（ollama） -------------------------- '''

# count = 1
# with open('./data/ENTITY/4/entity_id2name.txt', 'r') as ttttfile, \
#         open('./data/ENTITY/4/ollama embeddings.txt', 'w') as vectorfile:
#     for i in ttttfile.readlines():
#         print(count)
#         name = i.strip().split('\t')[1]
#         print(name)
#         if name == 'None':
#             vectorfile.write(str(count) + '\t' + 'None' + '\n')
#             count += 1
#             continue
#         ollama_embeddings = ollama.embeddings(model='llama2', prompt=f'{name}')
#         vectorfile.write(str(count) + '\t' + str(ollama_embeddings['embedding']) + '\n')
#         count += 1

''' -------------------------- 以上 -------------------------- '''


''' -------------------------- 提取相似度 -------------------------- '''

# count = 8725
# with open('./data/ENTITY/3/ollama embeddings.txt', 'r') as embeddingsfile:
#     for i in embeddingsfile.readlines():
#         vec1 = eval(i.strip().split('\t')[1])
#         if vec1 == None:
#             count += 1
#             with open(f'./data/ENTITY/3/similarity/{count}similarity.txt', 'w') as file:
#                 file.write('[]')
#             continue
#         list_similarity = []
#         with open('./data/ENTITY/3/ollama embeddings.txt', 'r') as file:
#             current = 8725
#             for j in file.readlines():
#                 if current == count:
#                     list_similarity.append('None')
#                 else:
#                     vec2 = eval(j.strip().split('\t')[1])
#                     if vec2 == None:
#                         list_similarity.append('None')
#                         current += 1
#                         continue
#                     similar = (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
#                     list_similarity.append(similar)
#                 current += 1
#         count += 1
#         print(count)
#         print(list_similarity)
#         with open(f'./data/ENTITY/3/similarity/{count}similarity.txt', 'w') as file:
#             file.write(str(list_similarity))


''' -------------------------- 以上 -------------------------- '''


''' -------------------------- 汇总相似度 -------------------------- '''

import os
import glob

dict = {}

def read_all_txt_files(directory):
    # 获取目录下所有的txt文件路径
    txt_files = glob.glob(os.path.join(directory, '*.txt'))

    i = 8726
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            print(f"Reading file: {txt_file}")
            content = eval(file.read())
            # 你可以在这里对内容进行处理
            j = 8726
            list = []
            for k in content:
                if k == 'None':
                    j += 1
                    continue
                if k > 0.9:
                    list.append(j)
                j += 1
            dict[i] = list
        i += 1

# 示例用法
directory = './data/ENTITY/3/similarity'  # 指定你的文件夹路径
read_all_txt_files(directory)

with open('./data/ENTITY/3/similarity.txt', 'w') as file:
    file.write(str(str(dict)))

print(dict)


''' -------------------------- 以上 -------------------------- '''