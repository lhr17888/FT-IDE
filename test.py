import torch

# import numpy as np
#
# # 加载npz文件
# data = np.load('tensors54.npz')
#
# # 读取每个张量
# tensor1 = data['tensor1']
# tensor2 = data['tensor2']
# tensor3 = data['tensor3']
# tensor4 = data['tensor4']
#
# # 设置打印选项，取消省略
# np.set_printoptions(threshold=np.inf)
#
# # 打印张量
# print(tensor1)
# print(tensor2)
# print(tensor3)
# print(tensor4)


# import gensim.downloader as api
# import numpy as np
# import ollama
# import pandas as pd
# import torch

# wv = api.load('word2vec-google-news-300')
# print('piano' in wv)
# print('Piano' in wv)
# vec_king = wv['University']
# vec_queen = wv['Guitar']

# a = torch.tensor([1.,2.,3.])
# b = torch.tensor([3.,6.,9.])
# dot_product = torch.dot(a, b)
# norm_i = torch.norm(a)
# norm_j = torch.norm(b)
# cos_sim = dot_product / (norm_i * norm_j)
# print(cos_sim>0.8)

# vec1 = ollama.embeddings(model='llama2',prompt="electronic guitar")['embedding'][:1000]
# vec2 = ollama.embeddings(model='llama2',prompt='guitar')['embedding'][:1000]
# print(len(vec2))
# print(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# with open('./mmm.txt', 'r') as file:
#     aaa = eval(file.read())
#     print(type(aaa))
#     print(aaa[1])
#     print(type(aaa[1]))
#     print(aaa)

# df = pd.read_csv('./ttt.txt', sep='\t', header=None)
# print(type(eval(df.iloc[1,1])))

# data = [[1., 2.], [3., 4.]]
# tensor_from_list = torch.tensor(data)
# huber_loss = torch.nn.HuberLoss(reduction='sum')
# print(huber_loss(tensor_from_list[0],tensor_from_list[1]))


# import os
# import glob
#
# dict = {}
#
# def read_all_txt_files(directory):
#     # 获取目录下所有的txt文件路径
#     txt_files = glob.glob(os.path.join(directory, '*.txt'))
#
#     i = 2910
#     for txt_file in txt_files:
#         with open(txt_file, 'r') as file:
#             print(f"Reading file: {txt_file}")
#             content = eval(file.read())
#             # 你可以在这里对内容进行处理
#             j = 2910
#             list = []
#             for k in content:
#                 if k == 'None':
#                     j += 1
#                     continue
#                 if k > 0.8:
#                     list.append(j)
#                 j += 1
#             dict[i] = list
#         i += 1
#
# # 示例用法
# directory = './data/ENTITY/1/similarity'  # 指定你的文件夹路径
# read_all_txt_files(directory)
#
# with open('./data/ENTITY/1/similartiry.txt', 'w') as file:
#     file.write(str(str(dict)))
#
# print(dict)

# bool_tensor = torch.tensor(True, dtype=torch.bool, device='cuda:0')
# print(bool_tensor)  # 输出: tensor(False, device='cuda:0')
#
# if bool_tensor:
#     print(111)

# if 7%2:
#     print(111)

# if not 21%5:
#     print(111)