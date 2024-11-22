# Selective Transfer Regression (STR)

## 簡介
Selective Transfer Regression (STR) 是一種用於回歸問題的機器學習模型，結合了支持向量回歸 (SVR) 和分布匹配技術。此模型特別適合於源數據和目標數據分布不同的情況。

## 最优化问题的推导过程

本文将详细推导如何将以下优化问题转化为标准的二次规划（Quadratic Programming, QP）形式：

$$
(\mathbf{w}, \mathbf{s}) = \arg\min_{\mathbf{w}, \mathbf{s}} R_{\mathbf{w}}(\mathcal{D}^{\text{tr}}, \mathbf{s}) + \lambda \Omega_{\mathbf{s}}(\mathbf{X}^{\text{tr}}, \mathbf{X}^{\text{te}})
$$

## 原始优化问题

定义优化目标为：

$$
\min_{\mathbf{s}} \quad \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(\mathbf{x}_i^{\text{tr}}) - \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(\mathbf{x}_j^{\text{te}})
$$

其中，\(\phi\) 是核函数，\(\mathbf{x}_i^{\text{tr}}\) 为训练集样本，\(\mathbf{x}_j^{\text{te}}\) 为测试集样本，\(n_{\text{tr}}\) 和 \(n_{\text{te}}\) 分别为训练集和测试集的样本数量，\(\mathbf{s}\) 是待优化的变量向量。

## 目标函数的展开

定义：

$$
\mathbf{u} = \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(\mathbf{x}_i^{\text{tr}}), \quad \mathbf{v} = \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(\mathbf{x}_j^{\text{te}})
$$

则优化目标函数可表示为：

$$
\min_{\mathbf{s}} \quad \| \mathbf{u} - \mathbf{v} \|^2
$$

展开目标函数：

$$
\| \mathbf{u} - \mathbf{v} \|^2 = (\mathbf{u} - \mathbf{v})^\top (\mathbf{u} - \mathbf{v}) = \mathbf{u}^\top \mathbf{u} - 2 \mathbf{u}^\top \mathbf{v} + \mathbf{v}^\top \mathbf{v}
$$

由于 \(\mathbf{v}\) 为常数向量，优化目标等价于：

$$
\mathbf{u}^\top \mathbf{u} - 2 \mathbf{u}^\top \mathbf{v}
$$

## 代入 \(\mathbf{u}\) 的表达式

代入 \(\mathbf{u}\) 的定义：

$$
\mathbf{u} = \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(\mathbf{x}_i^{\text{tr}})
$$

因此，

$$
\mathbf{u}^\top \mathbf{u} = \left( \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(\mathbf{x}_i^{\text{tr}}) \right)^\top \left( \frac{1}{n_{\text{tr}}} \sum_{j=1}^{n_{\text{tr}}} s_j \phi(\mathbf{x}_j^{\text{tr}}) \right) = \frac{1}{n_{\text{tr}}^2} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{tr}}} s_i s_j \phi(\mathbf{x}_i^{\text{tr}})^\top \phi(\mathbf{x}_j^{\text{tr}})
$$

$$
\mathbf{u}^\top \mathbf{v} = \left( \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(\mathbf{x}_i^{\text{tr}}) \right)^\top \left( \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(\mathbf{x}_j^{\text{te}}) \right) = \frac{1}{n_{\text{tr}} n_{\text{te}}} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{te}}} s_i \phi(\mathbf{x}_i^{\text{tr}})^\top \phi(\mathbf{x}_j^{\text{te}})
$$

## 引入核函数

定义核函数：

$$
k(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x})^\top \phi(\mathbf{y})
$$

则：

$$
\mathbf{u}^\top \mathbf{u} = \frac{1}{n_{\text{tr}}^2} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{tr}}} s_i s_j k(\mathbf{x}_i^{\text{tr}}, \mathbf{x}_j^{\text{tr}}) = \frac{1}{n_{\text{tr}}^2} \mathbf{s}^\top \mathbf{K} \mathbf{s}
$$

其中，\(\mathbf{K}\) 为训练集之间的核矩阵，定义为：

$$
K_{ij} = k(\mathbf{x}_i^{\text{tr}}, \mathbf{x}_j^{\text{tr}})
$$

$$
\mathbf{u}^\top \mathbf{v} = \frac{1}{n_{\text{tr}} n_{\text{te}}} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{te}}} s_i k(\mathbf{x}_i^{\text{tr}}, \mathbf{x}_j^{\text{te}})
$$

定义：

$$
\mathbf{k}_{\text{sum}} = \sum_{j=1}^{n_{\text{te}}} \mathbf{k}_j = \sum_{j=1}^{n_{\text{te}}} k(\mathbf{x}_i^{\text{tr}}, \mathbf{x}_j^{\text{te}})
$$

则，

$$
\mathbf{u}^\top \mathbf{v} = \frac{1}{n_{\text{tr}} n_{\text{te}}} \mathbf{s}^\top \mathbf{k}_{\text{sum}}
$$

## 构建优化目标的二次形式

将上述表达式代入优化目标：

$$
\min_{\mathbf{s}} \quad \frac{1}{n_{\text{tr}}^2} \mathbf{s}^\top \mathbf{K} \mathbf{s} - 2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} \mathbf{s}^\top \mathbf{k}_{\text{sum}}
$$

为了将其转化为标准的二次规划形式 \(\frac{1}{2} \mathbf{s}^\top \mathbf{P} \mathbf{s} + \mathbf{q}^\top \mathbf{s}\)，调整系数：

$$
\min_{\mathbf{s}} \quad \frac{1}{2} \cdot \frac{2}{n_{\text{tr}}^2} \mathbf{s}^\top \mathbf{K} \mathbf{s} + \left( -2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} \mathbf{k}_{\text{sum}} \right)^\top \mathbf{s}
$$

因此，

$$
\mathbf{P} = \frac{2}{n_{\text{tr}}^2} \mathbf{K}
$$

$$
\mathbf{q} = -2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} \mathbf{k}_{\text{sum}}
$$

## 重新定义目标函数参数

为了与标准形式对齐，令：

$$
\boldsymbol{\kappa} = \frac{n_{\text{tr}}}{n_{\text{te}}} \mathbf{k}_{\text{sum}}
$$

则，

$$
\mathbf{q} = -2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} \mathbf{k}_{\text{sum}} = -\frac{2}{n_{\text{tr}} n_{\text{te}}} \mathbf{k}_{\text{sum}} = -\frac{2}{n_{\text{tr}}^2} \cdot \frac{n_{\text{tr}}}{n_{\text{te}}} \mathbf{k}_{\text{sum}} = -\frac{2}{n_{\text{tr}}^2} \boldsymbol{\kappa}
$$

简化后，目标函数可以表示为：

$$
\min_{\mathbf{s}} \quad \frac{1}{2} \mathbf{s}^\top \mathbf{K} \mathbf{s} - \boldsymbol{\kappa}^\top \mathbf{s}
$$

其中，

$$
\boldsymbol{\kappa} = \frac{n_{\text{tr}}}{n_{\text{te}}} \mathbf{k}_{\text{sum}} = \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \mathbf{k}_j
$$

且 \(\mathbf{k}_j\) 的第 \(i\) 个元素为：

$$
k_i = k(\mathbf{x}_i^{\text{tr}}, \mathbf{x}_j^{\text{te}}), \quad \forall j = 1, \dots, n_{\text{te}}
$$

具体来说，

$$
\kappa_i = \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k(\mathbf{x}_i^{\text{tr}}, \mathbf{x}_j^{\text{te}}), \quad i = 1, \dots, n_{\text{tr}}
$$

## 总结

通过上述推导，我们成功地将原始的优化问题转化为标准的二次规划形式：

$$
\min_{\mathbf{s}} \quad \frac{1}{2} \mathbf{s}^\top \mathbf{K} \mathbf{s} - \boldsymbol{\kappa}^\top \mathbf{s}
$$

其中，

$$
\kappa_i := \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k(\mathbf{x}_i^{\text{tr}}, \mathbf{x}_j^{\text{te}}), \quad i = 1, \dots, n_{\text{tr}}
$$

这种形式便于利用二次规划求解器（如 `cvxopt`）进行高效求解。

## 功能
- **支持向量回歸 (SVR)**：使用 SVR 進行回歸分析。
- **分布匹配**：通過優化樣本權重來匹配源和目標數據的分布。
- **可視化**：提供模型決策邊界和數據點的可視化功能。

## 安裝
請按照以下步驟安裝和配置此項目：

1. 克隆此存儲庫：
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   ```
2. 進入項目目錄：
   ```bash
   cd yourproject
   ```
3. 安裝所需的依賴：
   ```bash
   pip install numpy scikit-learn cvxopt matplotlib
   ```

## 使用方法
提供一些基本的使用範例和說明：

### Selective Transfer Regression (STR)

#### 初始化模型

```python
from STR import SelectiveTransferRegression

str_model = SelectiveTransferRegression(
    C=10.0, 
    epsilon=0.1, 
    kernel='linear', 
    lambda_reg=1.0, 
    max_iter=5, 
    stm_epsilon=0.5, 
    B=5.0
)
```

#### 訓練模型

```python
str_model.fit(X_train, y_train, X_test, y_test)
```

- `X_train`: 訓練數據特徵，必須是 2D 陣列。
- `y_train`: 訓練數據標籤，必須是 1D 陣列。
- `X_test`: 測試數據特徵，必須是 2D 陣列。
- `y_test`: 測試數據標籤，必須是 1D 陣列。

#### 預測

```python
predictions = str_model.predict(X_new)
```

- `X_new`: 新的測試數據特徵，必須是 2D 陣列。

### Selective Transfer Machine (STM)

#### 簡介
Selective Transfer Machine (STM) 是一種用於分類問題的機器學習模型，基於支持向量機 (SVM) 和分布匹配技術。STM 專注於在源數據和目標數據之間進行有效的樣本權重調整，以提高分類性能。

#### 功能
- **支持向量機 (SVM)**：使用 SVM 進行分類分析。
- **分布匹配**：通過優化樣本權重來匹配源和目標數據的分布。
- **可視化**：提供模型決策邊界和數據點的可視化功能。

#### 初始化模型

```python
from STM import SelectiveTransferMachine

stm_model = SelectiveTransferMachine(
    C=10.0, 
    kernel='linear', 
    lambda_reg=1.0, 
    max_iter=10, 
    stm_epsilon=0.01, 
    B=1.0
)
```

#### 訓練模型

```python
stm_model.fit(X_train, y_train, X_test, y_test)
```

- `X_train`: 訓練數據特徵，必須是 2D 陣列。
- `y_train`: 訓練數據標籤，必須是 1D 陣列。
- `X_test`: 測試數據特徵，必須是 2D 陣列。
- `y_test`: 測試數據標籤，必須是 1D 陣列。

#### 預測

```python
predictions = stm_model.predict(X_new)
```

- `X_new`: 新的測試數據特徵，必須是 2D 陣列。

### 可視化

訓練過程中，模型會自動生成一個圖形，顯示 STR 和 STM 的決策邊界以及數據點。

## 範例

在 `STR.py` 和 `STM.py` 文件中提供了一個簡單的範例，展示了如何生成合成數據並訓練 STR 和 STM 模型。

## 貢獻

歡迎對此項目進行貢獻！請提交 pull request 或報告問題。

## 授權

此項目基於 MIT 許可證。

## 參考資料

- CHU, Wen-Sheng; DE LA TORRE, Fernando; COHN, Jeffery F. [Selective transfer machine for personalized facial action unit detection](https://openaccess.thecvf.com/content_cvpr_2013/papers/Chu_Selective_Transfer_Machine_2013_CVPR_paper.pdf). In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2013. p. 3515-3522.
