import numpy as np 
import matplotlib.pyplot as plt 
import plot_set
from sklearn.datasets import make_blobs

np.random.seed()

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0, cluster_std=0.5)

outlier = np.where(X[:,1] > 3)[0]

removed = outlier[10:80]
X = np.delete(X, removed, axis=0)
new_out = np.where(X[:, 1] > 3)[0]
inlier = np.where(X[:,1] <= 3)[0]

Inlier = X[inlier]
Outlier = X[new_out]



plt.hlines(y = 3, xmin = 0, xmax=1.5)

plt.vlines(x = 1.5, ymin = 3, ymax=5)
plt.scatter(Outlier[:,0],Outlier[:,1], color="y", label="Outliers")
plt.scatter(Inlier[:,0],Inlier[:,1], color="k", label="Inliers")
plt.ylim(0, 5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Classification example with correct target model")

plt.savefig("../figures/correct_class.pdf")

plt.show()


plt.hlines(y = 3, xmin = 3, xmax=4)

plt.vlines(x = 3, ymin = 3, ymax=5)
plt.scatter(Outlier[:,0],Outlier[:,1], color="y", label="Outliers")
plt.scatter(Inlier[:,0],Inlier[:,1], color="k", label="Inliers")
plt.ylim(0, 5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.title("Classification example with wrong target model")

plt.savefig("../figures/wrong_class.pdf")

plt.show()