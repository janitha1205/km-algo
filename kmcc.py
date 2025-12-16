import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=500, n_features=2, centers=6, random_state=23)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:, 0], X[:, 1])
plt.show()


class kmeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.clusters = {}
        np.random.seed(23)
        for i in range(self.k):
            center = 2 * (2 * np.random.random((self.data.shape[1],)) - 1)
            points = []
            clus = {"center": center, "points": points}
            self.clusters[i] = clus

    def distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def assign_clusters(self):
        for idx in range(self.data.shape[0]):
            dist = []

            curr_x = self.data[idx]

            for i in range(self.k):
                dis = self.distance(curr_x, self.clusters[i]["center"])
                dist.append(dis)
            curr_cluster = np.argmin(dist)
            self.clusters[curr_cluster]["points"].append(curr_x)

    def update_clusters(self):
        for i in range(self.k):
            points = np.array(self.clusters[i]["points"])
            if points.shape[0] > 0:
                new_center = points.mean(axis=0)
                self.clusters[i]["center"] = new_center

                self.clusters[i]["points"] = []

    def pred_cluster(self):
        pred = []
        for i in range(self.data.shape[0]):
            dist = []
            for j in range(self.k):
                dist.append(self.distance(self.data[i], self.clusters[j]["center"]))
            p = np.argmin(dist)
            pred.append(p)
            self.clusters[p]["points"] = self.data[i]
        return pred


algo = kmeans(3, X)
algo.assign_clusters()
algo.update_clusters()
pred = algo.pred_cluster()


plt.scatter(X[:, 0], X[:, 1], c=pred)
for i in range(algo.k):
    center = algo.clusters[i]["center"]
    plt.scatter(center[0], center[1], marker="^", c="red")
plt.show()
