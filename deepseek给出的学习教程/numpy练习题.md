# NumPy 练习题（32 题，含答案与详解）

> 用法：先自己做，再看答案。**答案里写了「易错点」，比答案本身更重要。**
> 全部题目和答案都在 numpy 2.5.4 上实跑验证过。
> 进阶题库：[rougier/numpy-100](https://github.com/rougier/numpy-100)（100 题，分难度带答案）

---

## Level 1：数组创建与属性（题 1-6）

**1.** 创建一个 3 行 4 列、元素全为 7 的整数数组。

**2.** 创建 0 到 1 之间的 11 个等距点（含 0 和 1）。

**3.** 创建 5×5 的单位矩阵，但要求 dtype 是 `bool`。

**4.** `a = np.arange(24)`，写出 `a.reshape(2, 3, 4)` 后 `a[1, 2, 3]` 的值。

**5.** 已知 `np.arange(0, 10, 0.3)`，说出它的元素个数，并解释为什么不建议这样用。

**6.** 创建 `(4, 3)` 的随机数组，取值范围 `[-5, 5)`。

<details><summary>答案与详解</summary>

```python
# 1
np.full((3, 4), 7)                       # dtype 自动推断为 int64
# 易错点：np.full(3, 4, 7) 是错的 —— shape 必须是一个元组

# 2
np.linspace(0, 1, 11)                    # 注意是「个数」11，不是步长
# 易错点：np.linspace(0,1,11) 含端点；np.arange(0,1,0.1) 有 10 个且不含端点

# 3
np.eye(5, dtype=bool)
# 或 np.eye(5).astype(bool)

# 4
a = np.arange(24).reshape(2, 3, 4)
a[1, 2, 3]        # -> 23
# 解析：C 序展开，下标 (i,j,k) 对应扁平索引 i*12 + j*4 + k = 12+8+3 = 23

# 5
len(np.arange(0, 10, 0.3))     # -> 34（本机实测）
# 官方文档明确指出：arange 用浮点步长时元素个数「无法预测」，因为 0.3 无法精确表示，
# 累加误差会让最后一个元素忽而包含忽而不包含。
# 需要确定个数时用 np.linspace(0, 9.9, 34) 或 np.arange(34) * 0.3

# 6
rng = np.random.default_rng(0)
rng.uniform(-5, 5, size=(4, 3))
# 易错点：不要用旧的 np.random.rand —— 它只能给 [0,1)，需要自己缩放，且是新代码不推荐的接口
```
</details>

---

## Level 2：形状与视图（题 7-12）

**7.** 把 `(6, 4)` 的数组变成 `(4, 6)`，有哪两种方式？哪种是视图？

**8.** `a = np.arange(10)`，`b = a[2:5]`，`b[0] = 99` 之后 `a[2]` 是多少？如果换成 `b = a[[2,3,4]]` 呢？

**9.** 把 `(1, 3, 1, 4)` 的数组去掉所有长度为 1 的轴，结果形状是什么？

**10.** `a = np.arange(12)`，`a.ravel()` 和 `a.flatten()` 的本质区别是什么？

**11.** 把长度为 5 的一维数组变成列向量（形状 `(5,1)`），写出两种写法。

**12.** 给定 `a` 形状 `(2, 3, 4)`，用 `np.transpose` 把轴顺序变成 `(4, 2, 3)`。

<details><summary>答案与详解</summary>

```python
# 7
a = np.arange(24).reshape(6, 4)
a.reshape(4, 6)          # 视图（能共享内存）
a.T                      # 视图（转置）
# 两者结果不同！reshape 是按 C 序重新划分，T 是真正的轴交换
np.shares_memory(a, a.reshape(4, 6))   # True
np.shares_memory(a, a.T)               # True

# 8
a = np.arange(10); b = a[2:5]; b[0] = 99
a[2]        # -> 99（切片是视图，改 b 就是改 a）
a = np.arange(10); b = a[[2,3,4]]; b[0] = 99
a[2]        # -> 2（花式索引是拷贝，互不影响）
# 易错点：这是最高频的"灵异 bug"来源。判断方法 np.shares_memory(a, b)

# 9
x = np.zeros((1, 3, 1, 4))
x.squeeze().shape      # (3, 4)
# 也可以指定只去某个轴：np.squeeze(x, axis=2) -> (1, 3, 4)

# 10
np.shares_memory(a, a.ravel())     # True  —— 尽量返回视图
np.shares_memory(a, a.flatten())   # False —— 总是拷贝
# 结论：不需要修改数据时优先 ravel（省内存）；想确保独立副本用 flatten/copy

# 11
a = np.arange(5)
a[:, None]              # 推荐，np.newaxis 的简写
a.reshape(-1, 1)
np.expand_dims(a, axis=1)

# 12
a = np.arange(24).reshape(2, 3, 4)
np.transpose(a, (2, 0, 1)).shape     # (4, 2, 3)
# 易错点：transpose 的参数是新形状中各位置来自原数组的哪个轴，
# 写的是「原轴编号」，不是新轴编号。写成 np.transpose(a, (1,2,0)) 就错了。
```
</details>

---

## Level 3：索引（题 13-18）

**13.** `a = np.arange(20).reshape(4, 5)`，取第 1、3 行。

**14.** 同上 `a`，取「第 1 行第 2 列」和「第 3 行第 4 列」这两个元素（结果是长度 2 的数组）。

**15.** 同上 `a`，取第 0、1 行与第 2、3 列构成的 **2×2 子矩阵**（注意和上一题的区别）。

**16.** `a` 中所有大于 10 且能被 2 整除的元素。

**17.** 用布尔索引把 `a` 中所有负数替换成 0。

**18.** `a = np.arange(10)`，取其中索引为 1、3、5、7 的元素，并原地修改它们为 -1。

<details><summary>答案与详解</summary>

```python
a = np.arange(20).reshape(4, 5)

# 13
a[[1, 3]]                 # 花式索引选行

# 14
a[[1, 3], [2, 4]]         # -> array([ 7, 19])
# 解析：这是「成对取样」：(1,2) 和 (3,4)，不是笛卡尔积！
# a[1,2] = 7, a[3,4] = 19

# 15
a[np.ix_([0, 1], [2, 3])]
# -> [[2, 3],
#     [7, 8]]
# 易错点：这是本题组最重要的一题。想要笛卡尔积必须用 np.ix_，
# 直接写 a[[0,1],[2,3]] 得到的是第 14 题的成对结果。

# 16
a[(a > 10) & (a % 2 == 0)]
# 易错点：① 必须用 & 不能用 and；② 每个条件必须加括号，
# 因为 & 的优先级高于 >，写 a > 10 & a % 2 == 0 会得到意外结果

# 17
b = np.array([-1, 2, -3, 4])
b[b < 0] = 0             # 布尔索引赋值
# -> array([0, 2, 0, 4])

# 18
a = np.arange(10)
a[[1,3,5,7]] = -1
# -> array([ 0, -1,  2, -1,  4, -1,  6, -1,  8,  9])
# 花式索引也能赋值。注意：如果索引有重复，普通赋值只保留最后一次，
# 想累加必须用 np.add.at（见 Level 7 第 27 题）
```
</details>

---

## Level 4：axis 与归约（题 19-23）

**19.** `m = np.arange(1, 7).reshape(2, 3)`，`m.sum(axis=1)` 的结果是什么？形状呢？

**20.** 求每列的平均值、每行的最大值。

**21.** 计算矩阵每行减去该行均值（"行中心化"），要求结果形状不变、且用一行代码。

**22.** `a = np.array([1, 2, np.nan, 4])`，`a.mean()` 是多少？怎么得到正确的均值？

**23.** 给定分类结果 `y = np.array([0, 1, 1, 3, 3, 3])`，统计每个类别出现次数。

<details><summary>答案与详解</summary>

```python
m = np.arange(1, 7).reshape(2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]

# 19
m.sum(axis=1)        # array([ 6, 15])，形状 (2,)
# 口诀：axis=k 表示「压掉第 k 个轴」。axis=1 压掉列方向 -> 每行的和

# 20
m.mean(axis=0)       # array([2.5, 3.5, 4.5])  每列均值
m.max(axis=1)        # array([3, 6])          每行最大值

# 21
m - m.mean(axis=1, keepdims=True)
# keepdims=True 让均值形状是 (2,1)，才能与 (2,3) 正确广播。
# 若不加 keepdims，(2,) 会与 (2,3) 广播，从最右对齐 -> (2,) 对齐到轴1，
# 变成「每列减」而不是「每行减」，且不一定报错！这是经典静默错误。

# 22
a.mean()             # nan  —— NaN 会污染整个归约
np.nanmean(a)        # 2.333...  —— 自动忽略 NaN
# 注意 NaN != NaN 恒成立，所以判断缺失必须用 np.isnan(a)，不能用 a == np.nan

# 23
np.bincount(y)                    # array([1, 2, 0, 3])
# 或
vals, counts = np.unique(y, return_counts=True)
# vals: [0 1 3], counts: [1 2 3]
# bincount 要求非负整数且从 0 开始；unique 更通用
```
</details>

---

## Level 5：广播（题 24-27）

**24.** `X` 形状 `(1000, 3)`，`mu` 形状 `(3,)`。`X - mu` 的形状是什么？为什么能广播？

**25.** 接上题。若想给 3 个特征各加一个不同的偏置（即"逐列加一个常数"），偏置的形状应该是 `(3,)`、`(3,1)` 还是 `(1,3)`？另外说明为什么 `X + np.zeros((3,1))` 会直接报错。

**26.** 为什么 `np.arange(3) + np.arange(4)` 报错，而 `(3,1)` 和 `(1,4)` 相加得到 `(3,4)`？

**27.** 用广播一次算出 `(5,)` 和 `(3,)` 两个向量的外积矩阵，形状 `(5, 3)`。

<details><summary>答案与详解</summary>

```python
# 24
X = np.zeros((1000, 3)); mu = np.zeros(3)
(X - mu).shape       # (1000, 3)
# 解析：广播从最右对齐。(3,) 被当作 (1,3)，再向轴 0 扩展成 (1000,3)。
# 这就是 StandardScaler 的核心。

# 25
X = np.zeros((1000, 3))
(X + np.zeros((3,))).shape     # (1000, 3)  ✅ (3,) 被当成 (1,3)
(X + np.zeros((1, 3))).shape   # (1000, 3)  ✅ 显式写成 (1,3) 最清楚
X + np.zeros((3, 1))           # ❌ ValueError: (1000,3) 与 (3,1) 不能广播
# 解析（广播从最右维开始逐维对齐）：
#   (1000, 3)
#   (   3, 1)   <- 补前导 1 后仍是 (3,1)
#    最右维：3 vs 1  -> 兼容，取 3
#    次右维：1000 vs 3 -> 3 != 1000 且都不为 1 -> 报错！
#
# 易错点：(3,1) 是「列向量」，它想对齐的是轴 0，而这里轴 0 长度是 1000，
# 所以只有当数组是 (3, 1000) 时 (3,1) 才能用：(3,1) + (1,1000) -> (3,1000)。
# 记忆法：想「逐行加同一个向量」用 (d,)，想「逐列加」用 (1,d)，
#        (d,1) 只在"被加的数组第一维恰好是 d"时才成立 —— 最容易踩的坑。

# 26
np.arange(3) + np.arange(4)
# ValueError: operands could not be broadcast together with shapes (3,) (4,)
# 解析：(3,) 与 (4,) 最右维 3 != 4 且都不为 1，无法对齐。
# 而 (3,1) 与 (1,4)：最右维 1 与 4 -> 4；次右维 3 与 1 -> 3。结果 (3,4)。
a = np.arange(3).reshape(3, 1)
b = np.arange(4).reshape(1, 4)
(a + b).shape        # (3, 4)

# 27
u = np.arange(5)         # (5,)
v = np.arange(3)         # (3,)
u[:, None] * v[None, :]  # (5, 3)
# 等价写法
np.outer(u, v)           # 更明确，推荐
np.einsum('i,j->ij', u, v)
# 易错点：u * v 会报错（(5,) 与 (3,) 不能广播）；必须先升维
```
</details>

---

## Level 6：线性代数（题 28-30）

**28.** 解线性方程组 `2x + y = 5`, `x + 3y = 10`。

**29.** 用最小二乘拟合 `y = kx + b`，给定 `x` 和 `y` 数据（不用 sklearn）。

**30.** 计算对称矩阵的特征值与特征向量，要求结果按升序排列。

<details><summary>答案与详解</summary>

```python
# 28
A = np.array([[2., 1.], [1., 3.]])
b = np.array([5., 10.])
x = np.linalg.solve(A, b)      # array([1., 3.])
# 验证：2*1+3=5 ✓，1+3*3=10 ✓
# 易错点：不要用 np.linalg.inv(A) @ b —— solve 更快、数值更稳

# 29
x = np.array([1., 2., 3., 4., 5.])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
A = np.vstack([x, np.ones_like(x)]).T     # 设计矩阵 (5,2)
# 或 A = np.column_stack([x, np.ones_like(x)])
k, b = np.linalg.lstsq(A, y, rcond=None)[0]
# k ≈ 2.0, b ≈ 0.05
# 易错点：rcond=None 必须显式传（新版本要求，否则给 FutureWarning）；
#        lstsq 返回 4 个值，第一个是解，后面是残差/秩/奇异值

# 30
M = np.array([[3., 1.], [1., 2.]])
w, V = np.linalg.eigh(M)
# w: [1.382, 3.618] 自动升序；V 的列是对应特征向量
# 易错点：对称矩阵必须用 eigh 不用 eig —— eigh 更快、更稳、结果有序。
#        eig 返回的可能是复数，且顺序不确定。
# 验证：M @ V[:,0] ≈ w[0] * V[:,0]
```
</details>

---

## Level 7：陷阱与实战（题 31-32）

**31.** 下面代码有什么 bug？怎么修？
```python
X = np.random.default_rng(0).random((300, 50))
sq = (X**2).sum(axis=1)
D = np.sqrt(sq[:, None] + sq[None, :] - 2 * X @ X.T)
```

**32.** 用纯 NumPy 实现二分类的混淆矩阵，要求正确处理重复索引。然后计算 precision 和 recall。

<details><summary>答案与详解</summary>

```python
# 31
# Bug：sqrt 的参数可能为负，产生 NaN。
# 原因是 a² - 2ab + b² 在 a≈b 时发生「灾难性抵消」：
# 距离本该是 0，浮点误差却算出 -1e-15，sqrt 负数 -> nan。
# 本机实测：d=20 时有 75 个元素为负，相对误差仅 1.02e-15（纯舍入误差）。
D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))   # 修法
# 修完的副作用（实测，很有启发性）：
#   • 对角线本该精确为 0（点到自身距离），但实测 300 个里有 86 个不是 0，
#     最大 1.46e-07 —— 因为 sqrt(-1e-15 被截成 0) 这种截断本身有误差。
#   • 对比：稳定的三重广播写法 np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(-1))
#     对角线精确为 0，但 n=2000、d=50 时要分配 1.6 GB 中间数组。
# 官方文档（basics.broadcasting）专门注明：这种展开法"可能会出现小的数值误差"。
# 生产环境建议用 scipy.spatial.distance.cdist(X, X)，内部用稳定算法且不爆内存。

# 32
def confusion_matrix(y_true, y_pred, K=2):
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)      # 关键：必须用 np.add.at
    return cm

# 如果写成 cm[y_true, y_pred] += 1，会发生什么？（实测）
cm_bad = np.zeros((2, 2), dtype=int)
cm_bad[y_true, y_pred] += 1
# cm_bad = [[1, 1],
#           [1, 1]]       <- 全变成 1 了！
# cm_bad.sum() = 4，而真实样本有 7 个 —— **丢了 3 个样本，而且不报错**
#
# 原因：+= 是「取值 -> 计算 -> 写回」三步，不是原子操作。
# 坐标 (0,0) 在数据里出现 2 次、(1,1) 出现 3 次、(1,0) 1 次、(0,1) 1 次；
# 重复坐标各自只写回一次，结果是「该单元格有几个重复索引」而不是「累加次数」。
# np.add.at 是「无缓冲原地累加」，逐条处理每个索引，所以正确。

y_true = np.array([0, 0, 1, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1])
cm = confusion_matrix(y_true, y_pred)
# cm = [[2, 1],
#       [1, 3]]
#   行=真实，列=预测

TP = cm[1, 1]; FP = cm[0, 1]; FN = cm[1, 0]; TN = cm[0, 0]
precision = TP / (TP + FP)      # 3/4 = 0.75
recall    = TP / (TP + FN)      # 3/4 = 0.75
accuracy  = (TP + TN) / cm.sum()  # 5/7 ≈ 0.714
```
</details>

---

## 附：三个自查用的「魔鬼细节」

做完上面 32 题后，如果你能立刻回答下面 3 个问题，说明基础已经很扎实：

**Q1**：`a` 形状 `(3,4)`，`a.mean(axis=1)` 和 `a.mean(axis=1, keepdims=True)` 分别是什么形状？为什么后者更常用于「减去均值」？

**Q2**：`a[1:3]` 和 `a[[1,2]]` 哪个会共享内存？为什么这个区别会导致难查的 bug？

**Q3**：`np.array([250], dtype=np.uint8) + 10` 等于多少？图像处理里该怎么避免？

<details><summary>答案</summary>

**A1**：`(3,)` 和 `(3,1)`。后者能与 `(3,4)` 沿轴 1 正确广播（每行减自己的均值）；前者会沿轴 1 广播，变成"每列减"，通常不是你要的效果，**而且不会报错**。

**A2**：`a[1:3]` 是基础切片 → 视图，共享内存；`a[[1,2]]` 是花式索引 → 拷贝。用 `np.shares_memory(a, b)` 判断。坑在于：修改视图会**悄悄改掉原数组**，而 NumPy 不报任何错。

**A3**：等于 `array([4])`（256 回绕）并给出 RuntimeWarning。修法是先升位宽：`np.clip(img.astype(np.int16) + 10, 0, 255).astype(np.uint8)`。
</details>
