# Selective Transfer Regression (STR)

## 最优化问题的推导过程

本文将详细推导如何将以下优化问题转化为标准的二次规划（Quadratic Programming, QP）形式：

$$
(w, s) = \arg\min_{w, s} R_{w}(\mathcal{D}^{\text{tr}}, s) + \lambda \Omega_{s}(X^{\text{tr}}, X^{\text{te}})
$$

## 原始优化问题

定义优化目标为：

$$
\min_{s} \quad \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(x_i^{\text{tr}}) - \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(x_j^{\text{te}})
$$

其中， $\phi$ 是核函数， $x_i^{\text{tr}}$ 为训练集样本， $x_j^{\text{te}}$ 为测试集样本， $n_{\text{tr}}$ 和 $n_{\text{te}}$ 分别为训练集和测试集的样本数量， $s$ 是待优化的变量向量。

## 目标函数的展开

定义：

$$
u = \frac{1}{n_{\text{tr}}} \sum_{i=1}^{n_{\text{tr}}} s_i \phi(x_i^{\text{tr}}), \quad v = \frac{1}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} \phi(x_j^{\text{te}})
$$

则优化目标函数可表示为：

$$
\min_{s} \quad \| u - v \|^2
$$

展开目标函数：

$$
\| u - v \|^2 = (u - v)^\top (u - v) = u^\top u - 2 u^\top v + v^\top v
$$

由于 $v$ 为常数向量，优化目标等价于：

$$
u^\top u - 2 u^\top v
$$

## 代入 $u$ 的表达式

代入 $u$ 的定义：

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

## 引入核函数

定义核函数：

$$
k(x, y) = \phi(x)^\top \phi(y)
$$

则：

$$
u^\top u = \frac{1}{n_{\text{tr}}^2} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{tr}}} s_i s_j k(x_i^{\text{tr}}, x_j^{\text{tr}}) = \frac{1}{n_{\text{tr}}^2} s^\top K s
$$

其中， $K$ 为训练集之间的核矩阵，定义为：

$$
K_{ij} = k(x_i^{\text{tr}}, x_j^{\text{tr}})
$$

$$
u^\top v = \frac{1}{n_{\text{tr}} n_{\text{te}}} \sum_{i=1}^{n_{\text{tr}}} \sum_{j=1}^{n_{\text{te}}} s_i k(x_i^{\text{tr}}, x_j^{\text{te}})
$$

定义：

$$
k_{\text{sum}} = \sum_{j=1}^{n_{\text{te}}} k_j = \sum_{j=1}^{n_{\text{te}}} k(x_i^{\text{tr}}, x_j^{\text{te}})
$$

则，

$$
u^\top v = \frac{1}{n_{\text{tr}} n_{\text{te}}} s^\top k_{\text{sum}}
$$

## 构建优化目标的二次形式

将上述表达式代入优化目标：

$$
\min_{s} \quad \frac{1}{n_{\text{tr}}^2} s^\top K s - 2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} s^\top k_{\text{sum}}
$$

为了将其转化为标准的二次规划形式 $\frac{1}{2} s^\top P s + q^\top s$，调整系数：

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

## 重新定义目标函数参数

为了与标准形式对齐，令：

$$
\kappa = \frac{n_{\text{tr}}}{n_{\text{te}}} k_{\text{sum}}
$$

则，

$$
q = -2 \cdot \frac{1}{n_{\text{tr}} n_{\text{te}}} k_{\text{sum}} = -\frac{2}{n_{\text{tr}} n_{\text{te}}} k_{\text{sum}} = -\frac{2}{n_{\text{tr}}^2} \cdot \frac{n_{\text{tr}}}{n_{\text{te}}} k_{\text{sum}} = -\frac{2}{n_{\text{tr}}^2} \kappa
$$

简化后，目标函数可以表示为：

$$
\min_{s} \quad \frac{1}{2} s^\top K s - \kappa^\top s
$$

其中，

$$
\kappa = \frac{n_{\text{tr}}}{n_{\text{te}}} k_{\text{sum}} = \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k_j
$$

且 $k_j$ 的第 $i$ 个元素为：

$$
k_i = k(x_i^{\text{tr}}, x_j^{\text{te}}), \quad \forall j = 1, \dots, n_{\text{te}}
$$

具体来说，

$$
\kappa_i = \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k(x_i^{\text{tr}}, x_j^{\text{te}}), \quad i = 1, \dots, n_{\text{tr}}
$$

## 总结

通过上述推导，我们成功地将原始的优化问题转化为标准的二次规划形式：

$$
\min_{s} \quad \frac{1}{2} s^\top K s - \kappa^\top s
$$

其中，

$$
\kappa_i := \frac{n_{\text{tr}}}{n_{\text{te}}} \sum_{j=1}^{n_{\text{te}}} k(x_i^{\text{tr}}, x_j^{\text{te}}), \quad i = 1, \dots, n_{\text{tr}}
$$

这种形式便于利用二次规划求解器（如 `cvxopt`）进行高效求解。
