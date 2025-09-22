import numpy as np

X = np.array([[1,1],[2,2],[3,3],[6,6],[7,7],[8,8]])
y = np.array(['A','A','A','B','B','B'])
q = np.array([4,4])

d = np.linalg.norm(X - q, axis=1)
k = 3
nearest = y[np.argsort(d)[:k]]
print("Predicted class:", max(set(nearest), key=list(nearest).count))
