from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

# 定义特征和目标变量
X = ... # 特征矩阵
y = ... # 目标变量

# 定义线性回归模型
lr = LinearRegression()

# 定义逐步回归选择器
sfs = SFS(lr, k_features=3, forward=True, floating=False, scoring='r2', cv=5)

# 训练逐步回归选择器
sfs.fit(X, y)

# 打印选定的特征
print('Selected features:', sfs.k_feature_idx_)
