import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from cvxopt import matrix, solvers

import matplotlib.pyplot as plt

class SelectiveTransferMachine:
    def __init__(self, C=1.0, kernel='linear', gamma='scale', lambda_reg=1.0, max_iter=10, stm_epsilon=0.01, B=1.0, visualization=False):
        """
        初始化 STM 類別
        :param C: SVM 的正則化參數
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
        self.visualization = visualization


    def _compute_weights(self, X_train, X_test, svc, y_train):
        """
        使用 QP 解決權重優化問題，結合分布匹配與分類損失
        """
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        
        # 計算核矩陣 (K) 和 kappa
        K_train = rbf_kernel(X_train, X_train)
        K_test = rbf_kernel(X_train, X_test)
        kappa = np.sum(K_test, axis=1) * n_train / n_test

        # 計算 hinge loss
        decision_function = svc.decision_function(X_train)
        hinge_loss = np.maximum(0, 1 - y_train * decision_function)  # hinge loss

        # 構建 QP 問題的參數
        n = len(hinge_loss)
        P = matrix(K_train)
        q = matrix((self.C / self.lambda_reg) * hinge_loss - kappa)

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
    
    
    # def __init__(self, C=1.0, kernel='linear', gamma='scale', lambda_reg=1.0, max_iter=10):
    #     """
    #     初始化 STM 類別
    #     :param C: SVM 的正則化參數
    #     :param kernel: SVM 的核函數類型（例如 'linear', 'rbf'）
    #     :param gamma: RBF 核函數的參數
    #     :param lambda_reg: 分布匹配損失的權重
    #     :param max_iter: 最大迭代次數
    #     """
    #     self.C = C
    #     self.kernel = kernel
    #     self.gamma = gamma
    #     self.lambda_reg = lambda_reg
    #     self.max_iter = max_iter


    # def _compute_weights(self, X_train, X_test, svc, y_train):
    #     """
    #     計算樣本權重，結合分布匹配與分類損失
    #     :param X_train: 訓練數據集
    #     :param X_test: 測試數據集
    #     :param svc: 已訓練的 SVM 分類器
    #     :param y_train: 訓練數據標籤
    #     :return: 樣本權重
    #     """
    #     n_train, n_test = X_train.shape[0], X_test.shape[0]
    #     K_train = rbf_kernel(X_train, X_train, gamma=self.gamma)
    #     K_test = rbf_kernel(X_train, X_test, gamma=self.gamma)

    #     # 計算分類損失（hinge loss）
    #     decision_function = svc.decision_function(X_train)
    #     hinge_loss = np.maximum(0, 1 - y_train * decision_function)  # hinge loss

    #     # 定義 QP 參數
    #     P = matrix(K_train)  # 核矩陣
    #     q = matrix((self.C / self.lambda_reg) * hinge_loss - np.sum(K_test, axis=1) * n_train / n_test)

    #     # 不等式約束: 0 <= s_i <= B (這裡設置 B=1)
    #     G = matrix(np.vstack([np.eye(n_train), -np.eye(n_train)]))
    #     h = matrix(np.hstack([np.ones(n_train), np.zeros(n_train)]))  # B=1, 下界=0

    #     # 等式約束: n_{tr}(1-ε) <= sum(s) <= n_{tr}(1+ε)
    #     epsilon = 0.01
    #     A = matrix(np.ones((1, n_train)))  # sum(s) 的係數
    #     b = matrix([n_train * (1 + epsilon), n_train * (1 - epsilon)])  # 上下界

    #     # 求解 QP 問題
    #     sol = solvers.qp(P, q, G, h, A, b)

    #     return np.array(sol['x']).flatten()  # 提取結果


    # def _compute_weights(self, X_train, X_test, svc, y_train):
    #     """
    #     計算樣本權重，結合分布匹配與分類損失
    #     :param X_train: 訓練數據集
    #     :param X_test: 測試數據集
    #     :param svc: 已訓練的 SVM 分類器
    #     :param y_train: 訓練數據標籤
    #     :return: 樣本權重
    #     """
    #     n_train, n_test = X_train.shape[0], X_test.shape[0]
    #     K_train = rbf_kernel(X_train, X_train, gamma=self.gamma)
    #     K_test = rbf_kernel(X_train, X_test, gamma=self.gamma)

    #     # 計算分類損失（hinge loss）
    #     decision_function = svc.decision_function(X_train)
    #     hinge_loss = np.maximum(0, 1 - y_train * decision_function)  # hinge loss
        
    #     # 分布匹配 + 損失影響
    #     s = np.linalg.solve(
    #         K_train + np.eye(n_train) * 1e-6,  # 加小數值防止奇異矩陣
    #         np.sum(K_test, axis=1) * n_train / n_test - self.lambda_reg * hinge_loss
    #     )
    #     return np.clip(s, 0, 1)  # 限制權重範圍


    def fit(self, X_train, y_train, X_test, y_test):
        """
        訓練 STM 模型
        :param X_train: 訓練數據特徵 array size (n, d)
        :param y_train: 訓練數據標籤 array size (n,)
        :param X_test: 測試數據特徵 array size (m, d)
        :param y_test: 測試數據標籤 array size (m,)
        """
        n_train = X_train.shape[0]
        self.weights = np.ones(n_train) / n_train  # 初始化權重

        for iteration in range(self.max_iter):
            # Step 1: 固定樣本權重，訓練加權 SVM
            svc = SVC(C=self.C, kernel=self.kernel)
            svc.fit(X_train, y_train, sample_weight=self.weights)

            # Step 2: 固定 SVM，更新樣本權重
            self.weights = self._compute_weights(X_train, X_test, svc, y_train)
            print(f"Iteration {iteration + 1}: Updated weights (min={self.weights.min():.4f}, max={self.weights.max():.4f})")

        self.svc = svc
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
        high_weight_mask = self.weights > np.percentile(self.weights, 95)  # 取權重前 5% 的樣本
        plt.scatter(X_train[~high_weight_mask & (y_train == 0)][:, 0], 
                    X_train[~high_weight_mask & (y_train == 0)][:, 1], 
                    color='blue', label='Train Class 0', marker='o')
        plt.scatter(X_train[~high_weight_mask & (y_train == 1)][:, 0], 
                    X_train[~high_weight_mask & (y_train == 1)][:, 1], 
                    color='darkblue', label='Train Class 1', marker='s', facecolors='none')
        plt.scatter(X_train[high_weight_mask & (y_train == 0)][:, 0], 
                    X_train[high_weight_mask & (y_train == 0)][:, 1], 
                    color='orange', label='Top 5% weight Class 0', marker='o', s=50)
        plt.scatter(X_train[high_weight_mask & (y_train == 1)][:, 0], 
                    X_train[high_weight_mask & (y_train == 1)][:, 1], 
                    color='orange', label='Top 5% weight Class 1', marker='s')
        
        # 測試資料
        plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label='Test Class 0', marker='o')
        plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='darkred', label='Test Class 1', marker='s', facecolors='none')

        # 繪製 STM 的 SVM 決策線
        xlim = plt.xlim()
        ylim = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
        Z = self.svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='b', levels=[1], alpha=0.5, linestyles='--')
        plt.contour(xx, yy, Z, colors='b', levels=[-1], alpha=0.5, linestyles='--')
        contour = plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles='--')
        plt.clabel(contour, inline=True, fontsize=8, fmt='STM Decision Boundary')

        # 繪製普通 SVM 的決策線
        normal_svc = SVC(C=self.C, kernel=self.kernel)
        normal_svc.fit(X_train, y_train)
        Z_normal = normal_svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_normal = Z_normal.reshape(xx.shape)
        contour_normal = plt.contour(xx, yy, Z_normal, colors='g', levels=[0], alpha=0.5, linestyles='-')
        plt.clabel(contour_normal, inline=True, fontsize=8, fmt='Normal SVM Decision Boundary')

        plt.title('STM and Normal SVM Decision Boundary and Data Points')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # 添加超参数的文本注释
        plt.text(0.05, 0.95, f'C: {self.C}\nB: {self.B}\nstm_epsilon: {self.stm_epsilon}', 
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        plt.legend()
        plt.show()


    def predict(self, X):
        """
        使用訓練好的 STM 模型進行預測
        :param X: 測試數據特徵
        :return: 預測標籤
        """
        return self.svc.predict(X)


# 測試 STM 模型
if __name__ == "__main__":
    
    # 創建合成數據
    # X_train, y_train = make_classification(n_samples=350, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=42)
    # X_test, y_test = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=42)
    X_train_label0 = np.random.uniform(1, 2, (200, 2))
    X_train_label1 = np.random.uniform(2, 3, (150, 2))
    X_train = np.concatenate([X_train_label0, X_train_label1])
    y_train = np.concatenate([np.zeros(200), np.ones(150)])

    _label0_x1 = np.random.uniform(2, 2.3, (50, 1))
    _label0_x2 = np.random.uniform(1.6, 1.9, (50, 1))
    X_test_label0 = np.concatenate([_label0_x1, _label0_x2], axis=1)

    _label1_x1 = np.random.uniform(2, 2.2, (50, 1))
    _label1_x2 = np.random.uniform(1.1, 1.4, (50, 1))
    X_test_label1 = np.concatenate([_label1_x1, _label1_x2], axis=1)

    X_test = np.concatenate([X_test_label0, X_test_label1])
    y_test = np.concatenate([np.zeros(50), np.ones(50)])

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # 初始化 STM 並訓練
    stm = SelectiveTransferMachine(C=10.0, kernel='linear', lambda_reg=1.0, max_iter=10, stm_epsilon=0, B=10.0, visualization=True)
    stm.fit(X_train, y_train, X_test, y_test)
