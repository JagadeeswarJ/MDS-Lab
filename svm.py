import numpy as np
from sklearn.svm import SVC

X = np.array([[2,2],[4,4],[4,0],[0,0],[0,2],[2,0]])
y = np.array([1,1,1,-1,-1,-1])

clf = SVC(kernel='linear', C=1e6)
clf.fit(X, y)

w, b = clf.coef_[0], clf.intercept_[0]
print("w =", w)
print("b =", b)
print("Equation: %.2f x1 + %.2f x2 + %.2f = 0" % (w[0], w[1], b))
