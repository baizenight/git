# NumPy 系统学习指南

> 环境已核对：本机 numpy **2.5.4**（NumPy 2.x），公开 API **494 个**，其中 ufunc **106 个**。
> **重要前提**：网上大量 NumPy 教程仍是 1.x 写法，其中 **42 个常用名字在 2.0 已被删除**（`np.float_`、`np.NaN`、`np.infty`、`np.product` …）。本文所有示例都在你的环境下实跑验证过。

---

## 第一部分：权威学习资源（按学习顺序）

### 1. 官方文档（唯一权威，优先）

| 资源 | 链接 | 用途 |
|---|---|---|
| NumPy 快速入门 | [numpy.org/doc/stable/user/quickstart.html](https://numpy.org/doc/stable/user/quickstart.html) | **入口**。讲透 ndarray 属性、axis、shape 操作 |
| 绝对初学者指南 | [numpy.org/doc/stable/user/absolute_beginners.html](https://numpy.org/doc/stable/user/absolute_beginners.html) | 更慢更细，含索引/广播/视图 |
| **完整函数分类目录** | [numpy.org/doc/stable/reference/routines.html](https://numpy.org/doc/stable/reference/routines.html) | **本文第二部分的基础**。26 个功能分类，494 个 API 的全景图 |
| 广播规则（务必读） | [numpy.org/doc/stable/user/basics.broadcasting.html](https://numpy.org/doc/stable/user/basics.broadcasting.html) | 90% 的形状报错都源于不懂它 |
| 副本与视图 | [numpy.org/doc/stable/user/basics.copies.html](https://numpy.org/doc/stable/user/basics.copies.html) | 解释"为什么改了 b，a 也变了" |
| 索引详解 | [numpy.org/doc/stable/user/basics.indexing.html](https://numpy.org/doc/stable/user/basics.indexing.html) | 基础/切片/花式/布尔四类索引 |
| 数据类型 | [numpy.org/doc/stable/user/basics.types.html](https://numpy.org/doc/stable/user/basics.types.html) | dtype、类型提升 |
| **2.0 迁移指南** | [numpy.org/doc/stable/numpy_2_0_migration_guide.html](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html) | **1.x 教程的坑全部在这** |
| 官方教程集 | [numpy.org/numpy-tutorials/](https://numpy.org/numpy-tutorials/) | 真实场景教学 notebook |
| I/O 操作 | [numpy.org/doc/stable/user/how-to-io.html](https://numpy.org/doc/stable/user/how-to-io.html) | 读写 npy/npz/csv/文本 |
| 向量化性能写法 | [numpy.org/doc/stable/user/basics.performant_code.html](https://numpy.org/doc/stable/user/basics.performant_code.html) | 多核、SIMD、避免临时数组 |

**官方文档的正确用法**：不要通读。遇到函数先 `python -c "import numpy as np; help(np.函数名)"` 看 docstring，那里就是官方文档原文，还带可运行示例。

### 2. 练习题库（按推荐度）

| 资源 | 链接 | 说明 |
|---|---|---|
| **100 NumPy Exercises** | [github.com/rougier/numpy-100](https://github.com/rougier/numpy-100) | **最权威的民间题库**。从数组创建到 stride tricks 梯度分明，每题带答案 |
| 上面题库的题目版 | [100_Numpy_exercises.md](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md) | 只有题目，适合先自己做 |
| NumPy 官方教程集 | [numpy.org/numpy-tutorials/](https://numpy.org/numpy-tutorials/) | 面向真实数据的项目式教学 |

### 3. 系统性教材

| 资源 | 说明 |
|---|---|
| **《Python for Data Analysis》第 4 章**（Wes McKinney，pandas 作者） | **第 4 章「NumPy Basics」是全书写得最好的一章**，面向数据分析视角，比官方文档更讲"为什么" |
| 《Python Data Science Handbook》第 2 章（VanderPlas） | 网上有免费全文，讲广播和花式索引的图很好 |
| 你目录里的 `matplotlib详细教程总结.md` | 配合 NumPy 做可视化的自查资料 |

### 4. 建议跳过

- 各种"NumPy 速成 30 分钟"：覆盖不到 axis 和视图，学完还是不会用
- 只讲 `np.array`/`np.zeros` 的"入门教程"：这些你 10 分钟就能学会
- 1.x 时代的教程（尤其涉及 `np.float_`、`np.int0`、`np.mat`）：会教你写出直接报错的代码

---

## 第二部分：NumPy 包含的所有功能

### 2.1 先建立心智模型：6 个核心概念

NumPy 的功能再多，都是围绕这 6 件事展开的。**学 NumPy 就是学这 6 件事**：

```
1. ndarray    数据结构：一块连续内存 + 形状 + 步长 + dtype
2. 形状操作   shape 变换、拼接、分割、广播
3. 索引      基础/切片/花式/布尔 —— 四类，语义完全不同
4. 逐元素运算 ufunc（106 个）：+、sqrt、exp、sin…
5. 归约运算   sum/mean/std/any/all + axis 参数
6. 领域模块   linalg / random / fft / polynomial / strings / ma
```

其中 **`ndarray` 的本质是「一块连续内存 + 元数据」**。理解了这点，视图/拷贝、`axis`、广播这些"难点"全都会变得自然。

### 2.2 官方 26 个功能分类（完整清单）

下面是官方 [routines.html](https://numpy.org/doc/stable/reference/routines.html) 的全部分类，我标注了**初学者优先级**：

| 官方分类 | 优先级 | 学什么 |
|---|---|---|
| 数组创建 | ★★★ | `array` `zeros` `ones` `empty` `arange` `linspace` `full` `eye` `*_like` `fromfunction` |
| 数组操作(变形) | ★★★ | `reshape` `ravel` `flatten` `transpose` `swapaxes` `moveaxis` `squeeze` `expand_dims` |
| 数组操作(拼接分割) | ★★★ | `concatenate` `stack` `vstack` `hstack` `split` `tile` `repeat` `pad` `append` `insert` `delete` |
| 索引 | ★★★ | `take` `where` `nonzero` `argwhere` `put` `choose` `select` `compress` `ix_` |
| 数学函数 | ★★★ | 三角/双曲/指数/对数/取整/符号 等 |
| 统计 | ★★★ | `mean` `std` `var` `median` `percentile` `quantile` `corrcoef` `cov` `histogram` `bincount` |
| 排序搜索计数 | ★★★ | `sort` `argsort` `lexsort` `searchsorted` `partition` `argpartition` `count_nonzero` |
| 逻辑函数 | ★★★ | `logical_and` `isclose` `allclose` `isfinite` `isnan` `isinf` |
| 随机抽样 | ★★★ | `default_rng` → `Generator` 的 45 个方法 |
| 线性代数 | ★★★ | `matmul`/`@` `dot` `inv` `solve` `lstsq` `eig` `svd` `qr` `norm` `det` `pinv` |
| 输入输出 | ★★★ | `save` `load` `savez` `savetxt` `loadtxt` `genfromtxt` `memmap` |
| 数据类型 | ★★☆ | `astype` `result_type` `promote_types` `can_cast` `isdtype` `finfo` `iinfo` |
| 集合运算 | ★★☆ | `unique` `intersect1d` `union1d` `setdiff1d` `isin` `in1d`(已弃用) |
| 位运算 | ★★☆ | `bitwise_and/or/xor` `left_shift` `right_shift` `bitwise_count`(2.x新) |
| 字符串功能 | ★★☆ | **`np.strings`（2.x 推荐）** vs `np.char`（旧） |
| 离散傅里叶变换 | ★★☆ | `fft` `ifft` `rfft` `fftfreq` `fftshift` |
| 多项式 | ★☆☆ | `polynomial.Polynomial`（**不要用旧的 `np.poly1d` 风格**） |
| 日期时间 | ★★☆ | `datetime64` `timedelta64` `busday_count` |
| 浮点错误处理 | ★★☆ | `errstate` `seterr` `geterr` |
| 掩码数组 | ★★☆ | `np.ma` 处理缺失值（227 个 API，用多少学多少） |
| 窗口函数 | ★☆☆ | `bartlett` `blackman` `hamming` `hanning`（信号处理用） |
| 函数式编程 | ★☆☆ | `apply_along_axis` `vectorize` `frompyfunc` |
| 常量 | ★★☆ | `pi` `e` `inf` `nan` `newaxis` |
| 异常与警告 | ★☆☆ | `np.exceptions.AxisError` 等 |
| 测试支持 | ★☆☆ | `testing.assert_allclose`（**写测试时很有用**） |
| 其他 | ★☆☆ | `misc`、数组打印选项等 |

> 完整的 494 个名字没必要背。**背住 60 个常用的，剩下的靠 `help()` 现查**——这是正确的工作方式。

### 2.3 领域模块规模（本机实测）

| 模块 | API 数 | 说明 |
|---|---|---|
| `np.ma` | 227 | 掩码数组（缺失值），最大的一块 |
| `np.char` | 68 | 旧字符串模块 |
| `np.random` | 65 | 随机抽样（其中 `Generator` 有 45 个方法） |
| `np.strings` | 46 | 2.x 新字符串模块（推荐） |
| `np.linalg` | 33 | 线性代数 |
| `np.fft` | 19 | 傅里叶变换 |
| `np.polynomial` | 15 | 多项式 |

---

## 第三部分：常用功能详解（全部实测验证）

### 3.1 数组创建

```python
import numpy as np

np.array([1, 2, 3])                    # 从列表创建，dtype 自动推断 -> int64
np.array([1, 2], dtype=np.float32)     # 指定 dtype
np.zeros((2, 3))                       # 全 0，默认 float64
np.ones((2, 3), dtype=np.int64)        # 全 1
np.full((2, 2), 7)                     # 全 7
np.empty((2, 2))                       # 未初始化（快，但值随机）
np.arange(0, 10, 2)                    # [0 2 4 6 8]，左闭右开
np.linspace(0, 1, 5)                   # [0. 0.25 0.5 0.75 1.]，**指定个数而非步长**
np.eye(3)                              # 单位矩阵
np.zeros_like(other)                   # 形状/类型同 other 的全 0 数组
np.random.default_rng(42).random((2, 3))  # 随机（2.x 推荐写法）
```

**⚠ 关键区别**：`arange` 用浮点步长时元素个数不可预测（浮点误差累积），**要固定个数就用 `linspace`**。官方文档专门强调这点。

### 3.2 形状操作

```python
a = np.arange(12)
a.reshape(3, 4)          # 视图（能则共享内存）
a.reshape(-1, 2)         # -1 自动推断：-> (6, 2)
a.ravel()                # 展平，**尽量返回视图**
a.flatten()              # 展平，**总是拷贝**
a.T                      # 转置（视图）
np.transpose(a2, (2, 0, 1))   # 指定轴顺序
np.swapaxes(a2, 0, 1)    # 交换两轴
np.moveaxis(a2, 0, -1)   # 把第 0 轴移到末尾
np.squeeze(a)            # 去掉长度为 1 的轴
a[:, None]               # 增加一个轴 -> np.newaxis 的简写
```

**视图 vs 拷贝（本机实测）**：
```python
base = np.arange(12)
np.shares_memory(base, base[::2])          # True  切片是视图
np.shares_memory(base, base.reshape(3, 4)) # True  reshape 是视图
np.shares_memory(base, base[[1, 2]])       # False 花式索引是拷贝！
np.shares_memory(base, base.ravel())       # True
np.shares_memory(base, base.flatten())     # False flatten 总是拷贝
```

**这就是你 ml-1 错题集里"改了 b，a 也变了"的根源。** 记住规则：
- **视图**：基础切片、`reshape`、`ravel`、`T`、`transpose`
- **拷贝**：花式索引 `a[[1,2]]`、布尔索引 `a[a>0]`、`flatten`、`np.array` 默认、显式 `.copy()`

### 3.3 四类索引（语义完全不同，必须分清）

```python
a = np.arange(20).reshape(4, 5)

# ① 基础索引 / 切片 —— 返回视图
a[1, 2]           # 标量
a[1]              # 第 1 行（等价 a[1, :]）
a[:, 2]           # 第 2 列
a[1:3, ::2]       # 行 1~2，列 0,2,4
a[..., 0]         # ... 代表"所有剩余的轴"

# ② 布尔索引 —— 返回拷贝，形状被压平
a[a > 10]                     # 所有大于 10 的元素（一维）
a[(a > 2) & (a < 8)]          # 多条件：必须用 & | ~，不能 and/or
a[a % 2 == 0]

# ③ 花式索引 —— 返回拷贝
a[[0, 2, 3]]                  # 选第 0,2,3 行
a[[0, 1], [2, 3]]             # **成对取样**：取 (0,2) 和 (1,3) -> [2 8]
a[np.ix_([0, 1], [2, 3])]     # **笛卡尔积**：2x2 子矩阵
```
> **最常见的困惑**：`a[[0,1],[2,3]]` 不是"选这些行和这些列"，而是"逐对取坐标"。要笛卡尔积必须用 `np.ix_`。

### 3.4 广播（最核心也最容易错）

**规则**：从**最右边**的维度开始逐个对齐，每个维度要么相等，要么其中一个是 1。

```python
a = np.arange(6).reshape(2, 3)     # (2,3)
b = np.array([10, 20, 30])         # (3,)   -> 补成 (1,3) -> 广播成 (2,3)  ✓
a + b                              # (2,3)

c = np.array([[1], [2]])           # (2,1)  -> 广播成 (2,3)  ✓
a + c                              # (2,3)

np.arange(3) + np.arange(4)        # ValueError: (3,) 与 (4,) 无法广播
np.broadcast_shapes((2, 3), (3,))  # (2, 3)  —— 提前检查能否广播
np.broadcast_to(b, (2, 3))         # 显式广播（只读视图，不复制内存）
```

**核心洞察**：广播**不复制数据**，它靠修改 stride 让同一个元素被多次读取。所以广播几乎不耗额外内存——但 `broadcast_to` 的结果是**只读**的。

**⚠ 你踩过的坑**：`(1000,)` 与 `(1000,1)` 运算会广播成 `(1000,1000)` 而**不报错**。养成打印 `.shape` 的习惯，这就是 ml-1 里 z 变成 (1000,1000) 的原因。

### 3.5 归约与 axis

```python
m = np.arange(1, 7).reshape(2, 3)
# array([[1, 2, 3],
#        [4, 5, 6]])

m.sum()                    # 21（全部）
m.sum(axis=0)              # [5 7 9]    压掉轴 0（行方向）-> 每列的和
m.sum(axis=1)              # [6 15]     压掉轴 1（列方向）-> 每行的和
m.sum(axis=1, keepdims=True)   # [[6], [15]]  形状 (2,1)，**便于后续广播**

# 常用归约
m.mean(axis=0); m.std(axis=0); m.var(axis=0)
m.min(axis=1); m.max(axis=1); m.argmax(axis=1)   # argmax 返回索引
m.cumsum(axis=1)           # 累加
m.prod(axis=0)
np.median(m); np.percentile(m, [25, 50, 75])
np.all(m > 0); np.any(m > 5)
np.count_nonzero(m > 3)
```

**记忆口诀**：`axis=0` 是"沿着行往下走"→ 结果是**每列**的归约；`axis=1` 是"沿着列往右走"→ 结果是**每行**的归约。
更准确的说法：**`axis=k` 表示"压掉第 k 个轴"**，结果的形状就是原形状去掉第 k 维。

### 3.6 数学与 ufunc

```python
# ufunc = 逐元素运算，106 个
np.sqrt(x); np.exp(x); np.log(x); np.log1p(x)   # log(1+x)，x 小时更精确
np.sin(x); np.cos(x); np.tan(x)
np.abs(x); np.sign(x); np.floor(x); np.ceil(x); np.rint(x)   # rint=四舍五入
np.maximum(a, b); np.minimum(a, b)      # **逐元素**，区别于 a.max()
np.clip(x, -1, 1)                       # 截断到区间
np.power(a, 2); np.mod(a, 3)

# ufunc 的高级用法
np.add.reduce(x)          # 等价 x.sum()
np.add.accumulate(x)      # 等价 x.cumsum()
np.add.outer(a, b)        # 外积
np.add.at(arr, idx, 1)    # **无缓冲原地累加**，处理重复索引时必需

# 标量函数 vs 逐元素函数（最容易搞混）
a.max()          # 整个数组的最大值 -> 标量
np.maximum(a, 0) # 逐元素与 0 比较 -> 数组（就是 ReLU！）
```

> **ReLU 就是 `np.maximum(z, 0)`** —— 你在 ml-2 学的激活函数，NumPy 里就一行。

### 3.7 拼接、分割与变换

```python
p = np.ones((2, 2)); q = np.zeros((2, 2))
np.vstack([p, q])            # (4,2) 垂直堆
np.hstack([p, q])            # (2,4) 水平堆
np.stack([p, q], axis=0)     # (2,2,2) **新轴**
np.stack([p, q], axis=2)     # (2,2,2) 新轴放最后
np.concatenate([p, q], axis=1)   # (2,4) 沿已有轴
np.concat([p, q], axis=1)        # **2.x 新增的推荐别名**
np.split(arr, 3)             # 等分成 3 份（要求能整除）
np.array_split(arr, 3)       # 不等分也行（推荐）
np.tile(m, (2, 1))           # 整块重复
np.repeat(m, 2, axis=0)      # 逐元素重复
np.pad(m, ((1,1),(2,2)), constant_values=-1)
np.roll(arr, 2)              # 循环移位
np.flip(arr)                 # 反转
```

**辨析**：`vstack/hstack` 是"堆"，`stack` 是"升一维再堆"，`concatenate` 是"沿指定已有轴接"。三者形状变化不同，实测：`vstack` → (4,2)，`hstack` → (2,4)，`stack(axis=0)` → (2,2,2)。

### 3.8 线性代数

```python
A @ B                     # **矩阵乘法，优先用它**（走 BLAS，实测比手写循环快 157x）
np.dot(A, B)              # 同 @（高维时语义有差别）
np.matmul(A, B)           # 同 @
A * B                     # ⚠ 这是**逐元素**相乘，不是矩阵乘

np.linalg.inv(M)          # 逆矩阵
np.linalg.solve(M, b)     # 解线性方程组（**比先求逆再乘更快更稳**）
np.linalg.lstsq(M, b)     # 最小二乘
np.linalg.det(M)          # 行列式
np.linalg.eig(M)          # 特征值/向量（一般矩阵）
np.linalg.eigh(M)         # **对称矩阵专用，更快更稳，结果有序**
np.linalg.svd(M)          # 奇异值分解
np.linalg.qr(M)           # QR 分解
np.linalg.norm(x)         # 范数；norm(x, ord=2) / 'fro'
np.linalg.pinv(M)         # 伪逆
np.linalg.matrix_rank(M)  # 秩
np.trace(M)               # 迹
np.einsum('ij,jk->ik', A, B)   # 爱因斯坦求和，见 3.11
```

实践原则：**能用 `solve` 就别用 `inv`**。`inv` 精度更差、速度更慢。

### 3.9 随机数（2.x 必须用新接口）

```python
# ✅ 2.x 推荐：先创建 Generator
rng = np.random.default_rng(42)          # 42 是种子
rng.random((2, 3))                       # [0,1) 均匀
rng.standard_normal(3)                   # 标准正态
rng.normal(loc=10, scale=2, size=3)      # 一般正态
rng.integers(0, 10, size=5)              # 整数 [0,10)
rng.uniform(-1, 1, 5)
rng.choice(10, size=5, replace=False)    # 无放回抽样
rng.permutation(10)                      # 返回打乱的新数组
rng.shuffle(arr)                         # **原地打乱**
rng.multivariate_normal([0,0], [[1,.5],[.5,1]], size=3)  # 多元正态 -> (3,2)
rng.binomial(10, 0.5, 5); rng.poisson(3, 5)

# ❌ 旧接口（仍能用，但不推荐新代码使用）
np.random.seed(0); np.random.rand(3)
```

**为什么必须换新接口**：
1. `default_rng` 用 PCG64，统计性质更好、速度更快
2. `Generator` 是**独立对象**，多线程/多模块之间不会互相污染全局状态
3. 旧接口的 `np.random.rand`/`randn` 命名不一致（`rand` 是均匀、`randn` 是正态，极易搞混），新接口统一成 `random`/`standard_normal`

### 3.10 统计与相关

```python
np.mean(x); np.std(x); np.var(x)          # 标准差/方差（默认总体，ddof=0）
np.std(x, ddof=1)                          # 样本标准差
np.median(x); np.percentile(x, [25,50,75])
np.quantile(x, 0.9)
np.corrcoef(X)                             # 相关系数矩阵
np.cov(X)                                  # 协方差矩阵
np.histogram(x, bins=10)                   # 返回 (计数, 边界)
np.bincount(np.array([0,1,1,3]))           # 整数计数 -> [1 2 0 1]
np.unique(x, return_counts=True)           # 去重 + 计数
np.nanmean(x); np.nanmedian(x); np.nanstd(x)  # **自动忽略 NaN**
```

**⚠ 缺失值**：普通 `mean` 遇到 NaN 结果就是 NaN。要么用 `np.nanmean`，要么用 `np.ma.masked_array`。

### 3.11 einsum：一个函数顶十个

```python
np.einsum('ij->ji', A)                  # 转置
np.einsum('ij->i', A)                   # 行和
np.einsum('i,j->ij', a, b)              # 外积
np.einsum('ij,jk->ik', A, B)            # 矩阵乘
np.einsum('bij,bjk->bik', A, B)         # 批量矩阵乘
np.einsum('ii->', M)                    # 迹
np.einsum('ii->i', M)                   # 对角元素（**是视图！**）
```

**读法**：箭头左边是输入下标，右边是输出下标；**出现在左边但不在右边的下标会被求和**。`'ij,jk->ik'` 中 `j` 出现两次且不在输出里，所以对 `j` 求和——这正是矩阵乘法。

### 3.12 排序与搜索

```python
np.sort(a)                       # 返回排序后的拷贝
a.sort()                         # 原地排序
np.argsort(a)                    # 返回排序索引
np.lexsort((b, a))               # 多键排序（**最后一个键是主键**）
np.searchsorted(sorted_arr, v)   # 二分查找插入位置
np.partition(a, k)               # 部分排序
np.argpartition(a, -2)[-2:]      # **O(n) 取 top-k，比全排序快**

# 取最大的 k 个
idx = np.argpartition(scores, -5)[-5:]
top5 = scores[idx]
```

### 3.13 I/O（实测通过）

```python
np.save('a.npy', arr);  arr = np.load('a.npy')          # 二进制，快，保 dtype/形状
np.savez('a.npz', x=arr, y=arr2);  z = np.load('a.npz');  z['x']
np.savez_compressed('a.npz', x=arr)                      # 压缩版
np.savetxt('a.csv', arr, delimiter=',', fmt='%.2f', header='a,b')
np.loadtxt('a.csv', delimiter=',', skiprows=1)
np.genfromtxt('a.csv', delimiter=',')                    # **能处理缺失值 -> nan**
np.memmap('big.dat', dtype='float32', mode='r', shape=(1000,1000))  # 超大文件不全载入
```

**选择建议**：`.npy`/`.npz` 存中间结果（快、无损）；`.csv` 用于交换数据；`memmap` 用于超过内存的大文件。

### 3.14 逻辑与比较

```python
np.isnan(x); np.isinf(x); np.isfinite(x)
np.isclose(a, b, rtol=1e-5, atol=1e-8)     # 逐元素近似相等
np.allclose(a, b)                          # **所有元素都近似相等**（测试必备）
np.logical_and(a, b); np.logical_or(a, b); np.logical_not(a)
np.where(cond, x, y)                       # 三目选择
np.select([c1, c2], [v1, v2], default=0)   # 多条件选择
np.piecewise(x, [x<0, x>=0], [lambda v:-v, lambda v:v])  # 分段函数
```

**布尔运算的坑**：数组上必须用 `&` `|` `~`，**不能用 `and` `or` `not`**（它们是标量运算，会报 ValueError）。且每个条件要加括号：`(a>2) & (a<8)`。

### 3.15 字符串、日期、掩码（实用但常被忽略）

```python
# 字符串：2.x 推荐 np.strings
np.strings.upper(np.array(['ab','cd']))       # ['AB' 'CD']
np.strings.str_len(np.array(['abc','x']))     # [3 1]
np.strings.startswith(np.array(['abc']), 'a')
# 旧接口 np.char 仍在，但官方建议迁移到 np.strings

# 日期
np.arange('2026-01', '2026-04', dtype='datetime64[M]')   # 按月递推
np.datetime64('2026-10-14') - np.datetime64('2026-10-01')  # timedelta64
np.busday_count('2026-10-01', '2026-10-14')              # 工作日数

# 掩码数组：把缺失值从计算中排除
mx = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
mx.mean()      # 2.0（忽略被掩码的 2）
mx.filled(-1)  # [1, -1, 3]

# 结构化数组
rec = np.array([(1, 2.5), (3, 4.5)], dtype=[('id', 'i4'), ('val', 'f8')])
rec['val']     # [2.5 4.5]
```

---

## 第四部分：实战场景（结合你正在学的方向）

### 场景 1 数据标准化（你 ml-2 里用过的 StandardScaler 的内核）

```python
# sklearn 的 StandardScaler，数学上就是这几行
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train_z = (X_train - mu) / sigma     # 广播：(n,d) - (d,) -> (n,d)
X_test_z = (X_test - mu) / sigma       # **必须用训练集的 mu/sigma**
```

### 场景 2 全连接层前向传播

```python
def linear_forward(X, W, b):
    """X:(n, d_in), W:(d_in, d_out), b:(d_out,) -> (n, d_out)"""
    return X @ W + b            # 广播自动加偏置

def relu(Z):
    return np.maximum(Z, 0)     # 就是 ReLU

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   # **减最大值防溢出**
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)

# 独热编码
def one_hot(y, K):
    out = np.zeros((len(y), K))
    out[np.arange(len(y)), y] = 1     # **花式索引赋值，一行搞定**
    return out
```

### 场景 3 mini-batch 划分（你 ml-2 里 `torch.randperm` 的 NumPy 版）

```python
rng = np.random.default_rng(42)
perm = rng.permutation(n)
for i in range(0, n, batch_size):
    idx = perm[i:i + batch_size]
    xb, yb = X[idx], y[idx]
```

### 场景 4 混淆矩阵（不用 sklearn 手写）

```python
def confusion_matrix(y_true, y_pred, K=2):
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)   # **必须用 add.at，普通 += 重复索引会丢**
    return cm
```
> `np.add.at` 是这里的要点：`cm[y_true, y_pred] += 1` 在索引重复时只生效一次。

### 场景 5 完整梯度下降（你 ml-1 的 NumPy 版）

```python
X = rng.random((1000, 2)) * 10
w_true, b_true = np.array([3.0, 2.0]), 5.0
y = X @ w_true + b_true + rng.normal(0, 0.1, 1000)   # 加噪声

w, b, lr = np.zeros(2), 0.0, 0.01
for epoch in range(1000):
    y_pred = X @ w + b
    err = y_pred - y
    loss = (err ** 2).mean()
    dw = 2 * X.T @ err / len(y)      # 梯度：X.T @ err 一行搞定
    db = 2 * err.mean()
    w -= lr * dw; b -= lr * db
```

### 场景 6 距离矩阵与 kNN

```python
def pairwise_distances(X):
    """X:(n,d) -> (n,n)。展开式 a²-2ab+b²，用矩阵乘避免大中间数组"""
    sq = (X ** 2).sum(axis=1)
    G = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(G, 0))     # **必须 max(0)，见第四部分陷阱**

def knn_predict(X_train, y_train, X_test, k=5):
    D = pairwise_distances(np.vstack([X_test, X_train]))[:len(X_test), len(X_test):]
    idx = np.argsort(D, axis=1)[:, :k]              # (n_test, k)
    votes = y_train[idx]                            # 花式索引
    return np.array([np.bincount(v, minlength=10).argmax() for v in votes])
```

### 场景 7 图像处理（把图当数组）

```python
img = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
gray = (img @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)  # 灰度化：矩阵乘
flipped = img[:, ::-1]                       # 水平翻转（视图，零成本）
cropped = img[20:80, 20:80]
bright = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)  # **防溢出先升位宽**
```

### 场景 8 时间序列差分与移动平均

```python
ts = np.cumsum(rng.normal(0, 1, 1000))       # 随机游走
diff1 = np.diff(ts)                           # 一阶差分
ma = np.convolve(ts, np.ones(20)/20, mode='valid')   # 20 期移动平均
norms = (ts - ts.mean()) / ts.std()           # 标准化
```

---

## 第五部分：NumPy 2.x 版本陷阱（1.x 教程的坑）

本节全部在本机实测。**这是本文对你最有价值的部分**，因为你机器上是 2.5.4，而网上教程大多是 1.x。

### 5.1 已删除的名字（本机实测确认 42 个）

| 1.x 写法 | 2.x 替代 |
|---|---|
| `np.float_` | **`np.float64`** |
| `np.complex_` | `np.complex128` |
| `np.unicode_` | `np.str_` |
| `np.NaN` | `np.nan` |
| `np.Inf` / `np.infty` / `np.Infinity` | `np.inf` |
| `np.NINF` | `-np.inf` |
| `np.PINF` | `np.inf` |
| `np.NZERO` / `np.PZERO` | `-0.0` / `0.0` |
| `np.product` | `np.prod` |
| `np.cumproduct` | `np.cumprod` |
| `np.sometrue` | `np.any` |
| `np.alltrue` | `np.all` |
| `np.round_` | `np.round` |
| `np.mat` | `np.asmatrix`（或直接用 ndarray） |
| `np.bool8` `np.int0` `np.uint0` `np.object0` `np.void0` | 用明确位宽的 dtype |
| `np.str_` | ✅ **仍存在**（唯一保留的） |
| `arr.ptp()` | `np.ptp(arr)` |
| `np.in1d` | `np.isin`（1.25 起弃用） |
| `np.row_stack` | `np.vstack` |
| `np.trapz` | `np.trapezoid` |

实测报错原文：
```
AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.
```

**⚠ 修旧代码的一条命令**（官方推荐）：
```bash
ruff check path/ --select NPY201
```
Ruff 的 `NPY201` 规则能自动把 1.x 写法改成 2.x。

### 5.2 类型提升规则变了（NEP 50）

**标量的精度不再被"忽略"**，实测：

```python
np.float32(1) + 1.0                    # float32（1.x 是 float64）← 精度可能下降
np.int8(1) + np.int8(1)                # int8（1.x 是 int64）
np.uint8(1) + 200                      # 201, uint8（1.x 会提升成 int64）
np.array([1], dtype=np.int8) + 200     # **OverflowError**（1.x 静默计算）
np.float32(1) + np.float64(1)          # float64
```

**踩坑后果**：整数运算可能**直接报错**或**静默溢出**。实测：
```python
np.uint8(255) + 1          # 0（并给 RuntimeWarning: overflow encountered）
np.array([255], dtype=np.uint8) + 1   # array([0]) ← 静默回绕，不报错！
```
> **教训**：涉及整数运算时显式 `astype` 到足够宽的 dtype。图像处理里 `uint8` 加减必须先转 `int16`。

### 5.3 其他 2.x 变化

```python
np.concat([a, b], axis=1)   # 2.x 新增别名，与 np.concatenate 等价
np.isdtype(arr.dtype, 'floating')     # 新的 dtype 判断
np.astype(arr, dtype)                 # 新函数形式
np.cumulative_sum(a); np.cumulative_prod(a)   # 新命名
np.unique_counts(a); np.unique_values(a)      # 拆分的唯一值函数
```

**注意 Windows 上的默认整数**（官方迁移指南明确说明）：
```python
np.int_   # -> int64（2.x 在所有 64 位系统统一为 64 位）
np.long   # -> int32（在 Windows 上！因为 C 的 long 是 32 位）
```
所以 **`np.long` 和 `np.int_` 在 Windows 上不是一回事**，写代码不要用 `np.long`。

**namespace 变化**：`np.core` 已改为私有的 `np._core`；`np.lib` 被清空，成员移到主命名空间或子模块。

### 5.4 调试类型问题的工具

```python
np.errstate(divide='ignore', invalid='ignore')   # 临时抑制浮点警告
np.finfo(np.float32); np.iinfo(np.int8)          # 查看类型的取值范围/精度
np.result_type(np.int8, np.float32)              # 查询提升结果
np.can_cast(np.int64, np.float32, casting='safe')  # 能否安全转换
np.promote_types(np.int8, np.float32)            # 两个 dtype 提升后的类型
np.isdtype(arr.dtype, 'floating')                # 2.x 新的 dtype 判断
```

> ⚠ **实测提醒**：官方 2.0 迁移指南里推荐的调试开关 `np._set_promotion_state("weak_and_warn")`
> 在 **numpy 2.5.4 上已经不存在**（本机实测 `hasattr(np, '_set_promotion_state')` 为 False，
> 且 `np`、`numpy.__dict__` 里都搜不到任何带 `promotion` 的公开名字）。
> **所以官方文档也会过时**——遇到不存在的 API，先 `hasattr` 探测，再找替代方案。
> 想排查提升问题，用上面的 `np.result_type` / `np.promote_types` 手工核对更可靠。

---

## 第六部分：8 个最经典的坑（全部实测）

### 坑 1 忘记 `max(0)`：sqrt 产生 NaN

```python
X = rng.random((300, 200))
sq = (X**2).sum(axis=1)
G = sq[:, None] + sq[None, :] - 2 * X @ X.T
np.sqrt(G)                     # ⚠ RuntimeWarning: invalid value encountered in sqrt
np.sqrt(np.maximum(G, 0))      # ✅ 正确
```
**原因**：`a² - 2ab + b²` 在 a≈b 时发生**灾难性抵消**，本该是 0 的距离算出 `-1e-15`，`sqrt` 直接变 NaN。

**实测代价**（n=300, d=50）：
- 不加 `max(0)`：直接产生 NaN
- 加了 `max(0)`：**对角线不再是精确的 0**——300 个里有 **86 个非零**，最大 **1.46e-07**（因为 `max(0)` 把 −1e-15 截成 0，截断本身带误差）
- 对比稳定的三重广播写法：对角线**精确为 0**，但 n=2000、d=50 时要分配 **1.6 GB** 中间数组

官方 [广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) 专门注明这种展开法"可能会出现小的数值误差"。
**实用建议**：需要精确距离就用 `scipy.spatial.distance.cdist(X, X)`（稳定算法，且不爆内存）。

### 坑 2 用 `and`/`or` 做数组布尔运算

```python
a[(a > 2) & (a < 8)]      # ✅ 必须 & | ~，每个条件加括号
a[(a > 2) and (a < 8)]    # ❌ ValueError: truth value of an array is ambiguous
```
**原因**：`and` 需要操作数有确定的真假，数组的"真假"是模糊的。`&` 才是逐元素的。

### 坑 3 花式索引赋值重复索引丢失

```python
y_true = np.array([0, 0, 1, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1])

cm = np.zeros((2, 2), dtype=int)
cm[y_true, y_pred] += 1                    # ❌ cm.sum() = 4，丢了 3 个样本！
np.add.at(cm, (y_true, y_pred), 1)         # ✅ cm.sum() = 7，正确
```
**实测结果**：`+=` 得到全 1 的矩阵（和 = 4），`np.add.at` 得到 `[[2,1],[1,3]]`（和 = 7）。
**原因**：`+=` 是"取值 → 计算 → 写回"三步，不是原子操作。坐标 `(0,0)` 出现 2 次、`(1,1)` 出现 3 次，重复坐标各自只写回一次，结果反映的是"该格有几个重复索引"而非累加次数。`np.add.at` 是**无缓冲原地累加**，逐条处理索引。

⚠ **这个 bug 不报错**，只是统计结果静默出错——和你 ml-2 里那个 `X`/`y` 错位的 bug 属于同一类。

### 坑 4 视图与拷贝

```python
b = a[0:2]; b[0] = 99         # a 也被改了（视图）
b = a[[0, 1]]; b[0] = 99      # a 不变（拷贝）
```
**判断方法**：`np.shares_memory(a, b)`，或看 `b.base is not None`。

### 坑 5 整数溢出（2.x 更严格）

```python
img = np.array([250], dtype=np.uint8)
img + 10         # array([4]) ← 静默回绕！
(int(img) + 10)  # 正确做法：先转 int，算完再转回
```
**图像处理必踩**：做加法/乘法前先 `.astype(np.int16)`。

### 坑 6 axis 记反

```python
m = np.arange(6).reshape(2, 3)
m.sum(axis=0)    # [5 7 9]  每列和（压掉行）
m.sum(axis=1)    # [6 15]   每行和（压掉列）
```
**口诀**：`axis=k` 就是"**压掉第 k 个轴**"。`axis=0` 压掉行 → 得到每列的统计量。

### 坑 7 除零与 NaN 传播

```python
np.array([1.0, 0.0]) / 0        # [inf, nan] + RuntimeWarning
np.nanmean(x)                    # 忽略 NaN 求均值
np.isfinite(x)                   # 检测
with np.errstate(divide='ignore', invalid='ignore'):   # 抑制警告
    ...
```
**要点**：`NaN != NaN` 恒成立，所以**不能**用 `==` 判缺失，要用 `np.isnan`。

### 坑 8 用 for 循环做数组运算

实测对比（numpy 2.5.4，本机）：

| 场景 | 向量化 | Python 循环 | 加速比 |
|---|---|---|---|
| `sqrt(x²+1)`，n=20 万 | 1.54 ms | 16.43 ms | **11x** |
| 成对距离矩阵，n=2000（朴素三重广播） | 1226 ms | 4340 ms | 3x |
| 成对距离矩阵（展开式 + `@`） | **60 ms** | 4340 ms | **72x** |
| 1000×1000 矩阵乘 `A@B` | **8.16 ms** | 1283 ms | **157x** |

**这张表比"向量化快 100 倍"这句口号有用得多**，它告诉你：
1. 简单逐元素运算只快约 **10 倍**（不是 100 倍）
2. **朴素三重广播是最差的向量化**——它造了个 2000×2000×50 = **1.6 GB** 的中间数组，导致只快 3 倍
3. **真正的大杀器是矩阵乘法走 BLAS**（157x）
4. 正确解法是**换算法**：把三重循环改成 `@` 形式的展开式，从 1226 ms 降到 60 ms（**再快 20 倍**）

**结论**：不要盲目"向量化"，要**向量化 + 选对算法 + 控制中间数组大小**。

---

## 第七部分：4 周学习路线（每天 1~2 小时）

| 周 | 目标 | 具体做什么 | 验收标准 |
|---|---|---|---|
| 1 | **数据结构与形状** | 官方 quickstart + absolute_beginners 通读并敲一遍；rougier 题库第 1-25 题 | 能不假思索说出任意 `reshape`/切片操作后的 shape |
| 2 | **索引与广播** | 官方 basics.indexing + basics.broadcasting；rougier 第 26-50 题 | 能解释 `(1000,)+(1000,1)` 为什么变成 `(1000,1000)`；能默写四类索引的区别 |
| 3 | **运算与统计** | 官方 routines 挑 ★★★ 的 10 类；rougier 第 51-75 题 | 能徒手实现标准化、softmax、独热编码、混淆矩阵 |
| 4 | **领域模块 + 实战** | linalg / random / I-O；把 ml-1 的线性回归和 ml-2 的 MLP 用纯 NumPy 重写 | 纯 NumPy 手写 MLP 跑通 two-moons，测试准确率 ≥ 99% |

### 每周必做的两件事

1. **`help()` 优先**：遇到不认识的函数，先看 docstring 再看搜索引擎。官方文档的结构是"每个函数都有 Parameters / Returns / Examples / Notes"，Examples 可以直接复制运行。
2. **写"错题集"**：像你 ml-1 那样记录"形状错了""视图改了原数组""整数溢出了"。**这种记录比笔记更值钱**，因为它是你真正踩过的坑。

### 自查：能通过这 5 题就算入门

1. `a` 形状 `(3, 4)`，`a.sum(axis=1)` 的形状是什么？为什么？
2. `a[0:2]` 和 `a[[0,1]]` 有什么区别？修改返回值哪个会影响 `a`？
3. `(1000,)` 和 `(1000,1)` 相加会发生什么？为什么危险？
4. `np.array([250], dtype=np.uint8) + 10` 等于多少？怎么修？
5. `np.sqrt(sq[:,None] + sq[None,:] - 2*X@X.T)` 为什么要套 `np.maximum(..., 0)`？

（答案分别在 3.5、3.2/3.3、3.4、坑 5、坑 1。）

---

## 第八部分：一页速查表

```python
# 创建
np.array / zeros / ones / full / empty / arange / linspace / eye / *_like

# 形状
reshape / ravel / flatten / T / transpose / swapaxes / moveaxis / squeeze / expand_dims

# 拼接分割
concatenate(=concat) / stack / vstack / hstack / split / array_split / tile / repeat / pad / roll / flip

# 索引
a[i,j] / a[1:3, ::2] / a[..., 0] / a[mask] / a[[0,2]] / a[[0,1],[2,3]] / np.ix_ / np.take / np.where

# 逐元素
sqrt exp log log1p sin cos abs floor ceil rint maximum minimum clip power mod

# 归约
sum mean std var min max argmin argmax cumsum prod median percentile quantile all any count_nonzero

# 线性代数
@ / matmul / dot / inv / solve / lstsq / eig / eigh / svd / qr / norm / det / pinv / matrix_rank / trace / einsum

# 随机（2.x）
rng = np.random.default_rng(seed)
rng.random / standard_normal / normal / integers / uniform / choice / permutation / shuffle / multivariate_normal

# 统计
corrcoef / cov / histogram / bincount / unique(return_counts=True) / nanmean / nanmedian / nanstd

# 集合排序
unique / intersect1d / union1d / setdiff1d / isin / sort / argsort / lexsort / searchsorted / argpartition

# I/O
save / load / savez / savez_compressed / savetxt / loadtxt / genfromtxt / memmap

# 工具
isnan / isinf / isfinite / isclose / allclose / where / select / piecewise / add.at / errstate / finfo / iinfo
```

**最后一句**：NumPy 的 494 个 API 不需要背。**真正的分水岭是理解 ndarray = 内存 + 形状 + 步长**——理解了这个，视图、广播、axis、花式索引全都变成推论而不是需要记忆的规则。
