# Selective Transfer Machine (STM)

## 最優化問題的推導過程

$$
(w, s) = \arg\min_{w, s} R_{w}(\mathcal{D}^{\text{tr}}, s) + \lambda \Omega_{s}(X^{\text{tr}}, X^{\text{te}})
$$

本文將詳細推導如何將以下優化問題轉化為標準的二次規劃（Quadratic Programming, QP）形式：

$$
\Omega_{s}(X^{\text{tr}}, X^{\text{te}}) = \left\| \frac{1}{n_{tr}} \sum_{i=1}^{n_{tr}} s_i \varphi(x_i^{tr}) - \frac{1}{n_{te}} \sum_{j=1}^{n_{te}} \varphi(x_j^{te}) \right\|_{\mathcal{H}}^2
$$

其中， $\phi$ 是核函數（kernel function）， $x_i^{\text{tr}}$ 為訓練集樣本， $x_j^{\text{te}}$ 為測試集樣本， $n_{\text{tr}}$ 和 $n_{\text{te}}$ 分別為訓練集和測試集的樣本數量， $s$ 是待優化的變量向量。

目標是將上述優化問題轉化為以下形式的二次規劃：

$$
\min_{s} \quad \frac{1}{2} s^\top K s - \kappa^\top s
$$

其中，

$$
\kappa_i := \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k(x_i^{\text{tr}}, x_j^{\text{te}}), \quad i = 1, \dots, n_{\text{tr}}
$$

## 步驟 1：定義優化目標

首先，明確原始優化問題的目標函數：

$$
\min_{s} \quad \left( \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(x_i^{\text{tr}}) - \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(x_j^{\text{te}}) \right)^2
$$

為了簡化符號，定義：

$$
u = \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(x_i^{\text{tr}}), \quad v = \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(x_j^{\text{te}})
$$

則目標函數可表示為：

$$
\min_{s} \quad \| u - v \|^2
$$

## 步驟 2：展開目標函數

將目標函數展開：

$$
\| u - v \|^2 = (u - v)^\top (u - v) = u^\top u - 2 u^\top v + v^\top v
$$

由於 $v$ 為常數向量，優化目標等價於最小化：

$$
u^\top u - 2 u^\top v
$$

忽略與 $s$ 無關的常數項 $v^\top v$。

## 步驟 3：代入 $u$ 的表達式

代入 $u$ 的定義：

$$
u = \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(x_i^{\text{tr}})
$$

因此，

$$
u^\top u = \left( \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(x_i^{\text{tr}}) \right)^\top \left( \frac{1}{n_{\text{tr}}} \sum_{j=1}^{n_{\text{tr}}} s_j \phi(x_j^{\text{tr}}) \right) = \frac{1}{n_{\text{tr}}^2} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{tr}}} s_i s_j \phi(x_i^{\text{tr}})^\top \phi(x_j^{\text{tr}}) 
$$

$$
u^\top v = \left( \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(x_i^{\text{tr}}) \right)^\top \left( \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(x_j^{\text{te}}) \right) = \frac{1}{n_{\text{tr}} n_{\text{te}}} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{te}}} s_i \phi(x_i^{\text{tr}})^\top \phi(x_j^{\text{te}}) 
$$

## 步驟 4：引入核函數

定義核函數：

$$
k(x, y) = \phi(x)^\top \phi(y)
$$

則：

$$
u^\top u = \frac{1}{n_{\text{tr}}^2} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{tr}}} s_i s_j k(x_i^{\text{tr}}, x_j^{\text{tr}}) = \frac{1}{n_{\text{tr}}^2} s^\top K s
$$

其中， $K$ 為訓練集之間的核矩陣，定義為：

$$
K_{ij} = k(x_i^{\text{tr}}, x_j^{\text{tr}})
$$

$$
u^\top v = \frac{1}{n_{\text{tr}} n_{\text{te}}} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{te}}} s_i k(x_i^{\text{tr}}, x_j^{\text{te}})
$$

定義：

$$
k_{\text{sum}} = \sum_{j=1}^{n_{\text{te}}} k_j = \sum_{j=1}^{n_{\text{te}}} k(x_i^{\text{tr}}, x_j^{\text{te}})
$$

則，

$$
u^\top v = \frac{1}{n_{\text{tr}} n_{\text{te}}} s^\top k_{\text{sum}}
$$

## 步驟 5：構建優化目標的二次形式

將上述表達式代入優化目標：

$$
\min_{s} \quad \frac{1}{n_{\text{tr}}^2} s^\top K s - 2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} s^\top k_{\text{sum}}
$$

為了將其轉化為標準的二次規劃形式 $\frac{1}{2} s^\top P s + q^\top s$，我們需要調整係數：

$$
\min_{s} \quad \frac{1}{2} \cdot \frac{2}{n_{\text{tr}}^2} s^\top K s + \left( -2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} k_{\text{sum}} \right)^\top s
$$

因此，

$$
P = \frac{2}{n_{\text{tr}}^2} K
$$

$$
q = -2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} k_{\text{sum}}
$$

## 步驟 6：重新定義目標函數參數

為了與標準形式對齊，令：

$$
\kappa = \frac{n_{\text{tr}}}{n_{\text{te}}} k_{\text{sum}}
$$

則，

$$
q = -2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} k_{\text{sum}} = -\frac{2}{n_{\text{tr}}} \cdot \frac{1}{n_{\text{te}}} k_{\text{sum}} = -\frac{2}{n_{\text{tr}}} \cdot \frac{n_{\text{te}}}{n_{\text{tr}}} \kappa = -\frac{2}{n_{\text{tr}}^2} \cdot \frac{n_{\text{tr}}}{n_{\text{te}}} k_{\text{sum}} = -\frac{2}{n_{\text{tr}} n_{\text{te}}} k_{\text{sum}}
$$

簡化後，目標函數可以表示為：

$$
\min_{s} \quad \frac{1}{2} s^\top K s - \kappa^\top s
$$

其中，

$$
\kappa = \frac{n_{\text{tr}}}{n_{\text{te}}} k_{\text{sum}} = \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k_j
$$

且 $k_j$ 的第 $i$ 個元素為：

$$
k_i = k(x_i^{\text{tr}}, x_j^{\text{te}}), \quad \forall j = 1, \dots, n_{\text{te}}
$$

具體來說，

$$
\kappa_i = \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k(x_i^{\text{tr}}, x_j^{\text{te}}), \quad i = 1, \dots, n_{\text{tr}}
$$

## 總結

通過上述推導，我們成功地將原始的優化問題轉化為標準的二次規劃形式：

$$
\min_{s} \quad \frac{1}{2} s^\top K s - \kappa^\top s
$$

其中，

$$
\kappa_i := \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k(x_i^{\text{tr}}, x_j^{\text{te}}), \quad i = 1, \dots, n_{\text{tr}}
$$

這種形式便於利用二次規劃求解器（如 `cvxopt`）進行高效求解。

---

# Selective Transfer Regression (STR) & Selective Transfer Machine (STM)

## 簡介

Selective Transfer Regression (STR) 和 Selective Transfer Machine (STM) 是兩種機器學習模型，旨在解決源數據和目標數據分布不同的問題。

- **Selective Transfer Regression (STR)**：用於回歸問題，結合支持向量回歸 (SVR) 和分布匹配技術。
- **Selective Transfer Machine (STM)**：用於分類問題，基於支持向量機 (SVM) 和分布匹配技術。


## Selective Transfer Regression (STR)

### 功能
- **支持向量回歸 (SVR)**：使用 SVR 進行回歸分析。
- **分布匹配**：通過優化樣本權重來匹配源和目標數據的分布。
- **可視化**：提供模型決策邊界和數據點的可視化功能。

### 初始化模型

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

### 訓練模型

```python
str_model.fit(X_train, y_train, X_test, y_test)
```

- `X_train`: 訓練數據特徵，必須是 2D 陣列。
- `y_train`: 訓練數據標籤，必須是 1D 陣列。
- `X_test`: 測試數據特徵，必須是 2D 陣列。
- `y_test`: 測試數據標籤，必須是 1D 陣列。

### 預測

```python
predictions = str_model.predict(X_new)
```

- `X_new`: 新的測試數據特徵，必須是 2D 陣列。


## Selective Transfer Machine (STM)

### 功能
- **支持向量機 (SVM)**：使用 SVM 進行分類分析。
- **分布匹配**：通過優化樣本權重來匹配源和目標數據的分布。
- **可視化**：提供模型決策邊界和數據點的可視化功能。

### 初始化模型

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

### 訓練模型

```python
stm_model.fit(X_train, y_train, X_test, y_test)
```

- `X_train`: 訓練數據特徵，必須是 2D 陣列。
- `y_train`: 訓練數據標籤，必須是 1D 陣列。
- `X_test`: 測試數據特徵，必須是 2D 陣列。
- `y_test`: 測試數據標籤，必須是 1D 陣列。

### 預測

```python
predictions = stm_model.predict(X_new)
```


- `X_new`: 新的測試數據特徵，必須是 2D 陣列。


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


## 可視化

訓練過程中，STR 和 STM 模型會自動生成圖形，顯示模型的決策邊界和數據點的分佈情況。


## 範例

在 `STR.py` 和 `STM.py` 文件中提供了簡單範例，展示了如何生成合成數據並訓練 STR 和 STM 模型。


## 貢獻

歡迎對此項目進行貢獻！請提交 pull request 或報告問題。


## 授權

此項目基於 MIT 許可證。
