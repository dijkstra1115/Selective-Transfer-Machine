"""
X_train and X_test must be 2D array (n, d)
y_train and y_test must be 1D array (n,)
"""


import numpy as np
from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

import matplotlib.pyplot as plt

class SelectiveTransferRegression:
    def __init__(self, C=1.0, epsilon=0.1, kernel='linear', gamma='scale', lambda_reg=1.0, max_iter=10, stm_epsilon=0.01, B=1.0, visualization=False):
        """
        初始化 STM 類別
        :param C: SVM 的正則化參數
        :param epsilon: ε-insensitive loss 的參數
        :param kernel: SVM 的核函數類型（例如 'linear', 'rbf'）
        :param gamma: RBF 核函數的參數
        :param lambda_reg: 分布匹配損失的權重
        :param max_iter: 最大迭代次數
        :param stm_epsilon: 控制權重總和約束的範圍
        :param B: 權重的上界
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.stm_epsilon = stm_epsilon
        self.B = B
        self.epsilon = epsilon
        self.visualization = visualization


    def _compute_weights(self, X_train, X_test, svr, y_train):
        """
        使用 QP 解決權重優化問題，結合分布匹配與分類損失
        !!!注意!!! y_train 需要是 1D array (n,) 而不能是 2D array (n, 1)
        """
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        
        # 計算核矩陣 (K) 和 kappa
        K_train = rbf_kernel(X_train, X_train)
        K_test = rbf_kernel(X_train, X_test)
        kappa = np.sum(K_test, axis=1) * n_train / n_test

        # 計算 ε-insensitive loss
        decision_function = svr.predict(X_train)
        epsilon_loss = np.maximum(0, np.abs(y_train - decision_function) - self.epsilon)

        # 構建 QP 問題的參數
        n = len(epsilon_loss)
        P = matrix(K_train)
        q = matrix((self.C / self.lambda_reg) * epsilon_loss - kappa)

        # 構建 G 和 h 矩陣
        G_lower = -np.eye(n)  # -s_i <= 0
        h_lower = np.zeros(n)

        G_upper = np.eye(n)   # s_i <= B
        h_upper = self.B * np.ones(n)

        G_sum_lower = -np.ones((1, n))  # -sum(s) <= -n_tr(1 - epsilon)
        h_sum_lower = -n_train * (1 - self.stm_epsilon)

        G_sum_upper = np.ones((1, n))   # sum(s) <= n_tr(1 + epsilon)
        h_sum_upper = n_train * (1 + self.stm_epsilon)

        # 合併所有 G 和 h
        G = np.vstack([G_lower, G_upper, G_sum_lower, G_sum_upper])
        h = np.hstack([h_lower, h_upper, h_sum_lower, h_sum_upper])

        # 將 G 和 h 轉換為 cvxopt 格式
        G = matrix(G)
        h = matrix(h)

        # 使用 cvxopt 求解 QP 問題
        sol = solvers.qp(P, q, G, h)

        # 提取權重解
        s = np.array(sol['x']).flatten()
        return s


    def fit(self, X_train, y_train, X_test, y_test):
        """
        訓練 STR 模型
        :param X_train: 訓練數據特徵 array size (n, d)
        :param y_train: 訓練數據標籤 array size (n,)
        :param X_test: 測試數據特徵 array size (m, d)
        :param y_test: 測試數據標籤 array size (m,)
        """
        n_train = X_train.shape[0]
        self.weights = np.ones(n_train) / n_train  # 初始化權重

        for iteration in range(self.max_iter):
            # Step 1: 固定樣本權重，訓練加權 SVR
            svr = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
            svr.fit(X_train, y_train, sample_weight=self.weights)

            # Step 2: 固定 SVR，更新樣本權重
            self.weights = self._compute_weights(X_train, X_test, svr, y_train)
            print(f"Iteration {iteration + 1}: Updated weights (min={self.weights.min():.4f}, max={self.weights.max():.4f})")

        self.svr = svr
        if self.visualization:
            self._visualize(X_train, y_train, X_test, y_test)


    def _visualize(self, X_train, y_train, X_test, y_test):
        """
        預測並繪製結果
        :param X_train: 訓練數據特徵
        :param y_train: 訓練數據標籤
        :param X_test: 測試數據特徵
        :param y_test: 測試數據標籤
        """

        # 繪製散點圖
        plt.figure(figsize=(10, 6))
        
        # 訓練資料
        high_weight_mask = self.weights > np.percentile(self.weights, 90)  # 取權重前 5% 的樣本
        plt.scatter(X_train[~high_weight_mask], y_train[~high_weight_mask], color='blue', label='Train', marker='o')
        plt.scatter(X_train[high_weight_mask], y_train[high_weight_mask], color='orange', label='Top 5% weight', marker='o', s=50)
        
        # 測試資料
        plt.scatter(X_test, y_test, color='red', label='Test', marker='o')

        # 繪製 STR 的決策線
        xlim = plt.xlim()
        xx = np.linspace(xlim[0], xlim[1], 100).reshape(-1, 1)
        yy = self.svr.predict(xx)
        plt.plot(xx, yy + self.epsilon, color='blue', linestyle='--', label='U/L Bound')
        plt.plot(xx, yy - self.epsilon, color='blue', linestyle='--')
        plt.plot(xx, yy, color='k', linestyle='-', label='STR')
        plt.fill_between(xx.ravel(), yy - self.epsilon, yy + self.epsilon, color='blue', alpha=0.1, label='Epsilon Tube')

        # 繪製普通 SVR 的決策線
        normal_svr = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        normal_svr.fit(X_train, y_train)
        yy_normal = normal_svr.predict(xx)
        plt.plot(xx, yy_normal, color='g', linestyle='-', label='SVR')

        plt.title('STR Decision Boundary and Data Points')
        plt.xlabel('Feature')
        plt.ylabel('Target')

        # 添加超参数的文本注释
        plt.text(0.05, 0.95, f'C: {self.C}\nB: {self.B}\nstm_epsilon: {self.stm_epsilon}', 
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        plt.legend()
        plt.show()


    def evaluate(self, X_test, y_test):
        """
        評估模型
        :param X_test: 測試數據特徵 array size (m, d)
        :param y_test: 測試數據標籤 array size (m,)
        """
        y_pred = self.predict(X_test)
        y_error = y_pred - y_test
        mean = np.mean(y_error)
        std = np.std(y_error)
        suc_13 = np.sum(np.abs(y_error) < 13) / len(y_error)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mae': mae,
            'mean_error': mean,
            'std_error': std,
            'suc_13': suc_13,
            'r2_score': r2,
            'error_correlation': np.corrcoef(y_test, y_error)[0, 1]
        }

        return metrics


    def predict(self, X):
        """
        使用訓練好的 STM 模型進行預測
        :param X: 測試數據特徵
        :return: 預測標籤
        """
        return self.svr.predict(X)


# 測試 STM 模型
if __name__ == "__main__":
    
    # 創建合成數據
    X_train = np.random.uniform(-2, 2, size=(200, 1))  # 生成 200 個隨機點
    y_train = (X_train**2 - 0.5 * X_train - 1.5 + np.random.normal(0, 0.5, size=(200, 1))).ravel() # 因為 y_train 需要是 1D array (n,) 而不能是 2D array (n, 1)

    # X_test = np.random.uniform(-1.7, -0.7, size=(50, 1))  # 生成 50 個隨機點
    # y_test = (-2.7 * X_test -2.6 + np.random.normal(0, 0.5, size=(50, 1))).ravel()

    X_test = np.random.uniform(0.5, 1, size=(50, 1))  # 生成 50 個隨機點
    y_test = np.random.uniform(-2, -3, size=(50, 1)).ravel() # 因為 y_test 需要是 1D array (n,) 而不能是 2D array (n, 1)

    # 初始化 STR 並訓練
    str = SelectiveTransferRegression(C=10.0, epsilon=0.1, kernel='rbf', lambda_reg=1.0, max_iter=5, stm_epsilon=0.5, B=5.0, visualization=True)
    str.fit(X_train, y_train, X_test, y_test)
    metrics = str.evaluate(X_test, y_test)
    print(metrics)
