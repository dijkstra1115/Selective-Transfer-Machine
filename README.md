# Selective Transfer Regression (STR)

## 簡介
Selective Transfer Regression (STR) 是一種用於回歸問題的機器學習模型，結合了支持向量回歸 (SVR) 和分布匹配技術。此模型特別適合於源數據和目標數據分布不同的情況。

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
