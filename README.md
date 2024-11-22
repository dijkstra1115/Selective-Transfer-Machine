# Selective Transfer Regression (STR)

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
