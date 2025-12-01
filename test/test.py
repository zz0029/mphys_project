import numpy as np
from sklearn.neighbors import NearestNeighbors

# 假装这是 RGZ 的 512D 特征，这里用 3 维、5 个样本演示
rgz_features_512 = np.array([
    [1.0, 0.0, 0.0],   # RGZ 0
    [0.0, 1.0, 0.0],   # RGZ 1
    [0.0, 0.0, 1.0],   # RGZ 2
    [1.0, 1.0, 0.0],   # RGZ 3
    [1.0, 0.0, 1.0],   # RGZ 4
])

# 假装这是 ORC 的特征：2 个 ORC
orc_features_512 = np.array([
    [1.0, 0.1, 0.0],   # ORC 0
    [0.0, 0.0, 1.0],   # ORC 1
])

K = 2  # 每个 ORC 找 2 个最近邻
nn = NearestNeighbors(n_neighbors=K, metric="cosine")
nn.fit(rgz_features_512)

# distances = 1 - cosine_similarity
distances, indices = nn.kneighbors(orc_features_512)

print("indices (每行是一个 ORC 的邻居 RGZ 索引):")
print(indices)

print("\ndistances (对应的 cosine 距离 = 1 - cos_sim):")
print(distances)

# 也可以把它们转回 cosine similarity 看：
cos_sims = 1.0 - distances
print("\ncosine similarities:")
print(cos_sims)