# from matplotlib import pyplot as plt
# import numpy as np
#
# if __name__ == "__main__":
#     m = np.linspace(1, 1000, 1000)
#     y = [(1 - (1 / example)) ** example for example in m]
#     y1 = [y[i] for i in range(len(y)) if y[i] < 0.36]
#     num = len(y1)
#     print(num)
#     plt.plot(m, y)
#     plt.show()

# from collections import Counter
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import numpy as np
#
#
# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))
#
#
# def knn(X_train, y_train, x_test, k):
#     distances = [euclidean_distance(x_test, x) for x in X_train]
#     k_indices = np.argsort(distances)[:k]
#     k_nearest_labels = [y_train[i] for i in k_indices]
#     most_common = Counter(k_nearest_labels).most_common(1)
#     return most_common[0][0]
#
#
# iris = load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234
# )
#
# predictions = [knn(X_train, y_train, x_test, k=3) for x_test in X_test]
# accuracy = np.sum(predictions == y_test) / len(y_test)
# print(f"Accuracy: {accuracy:.2f}")

import scipy.optimize as opt
import scipy.stats as st
import numpy as np


def kumaraswamy_logL(log_par, data):
    N = len(data)
    a, b = np.exp(log_par)
    logL = (
        N * np.log(a * b)
        + (a - 1) * np.sum(np.log(data))
        + (b - 1) * np.sum(np.log(1 - np.power(data, a)))
    )
    return logL


def kumaraswamy_mle(data):
    res = opt.minimize(
        fun=lambda log_params, data: -kumaraswamy_logL(log_params, data),
        x0=np.array([0.5, 0.5]),
        args=(data,),
        method="BFGS",
    )
    a, b = np.exp(res.x)
    return a, b
