# Matplotlib 详细使用教程总结

> 本文档由网络资料（以官方文档为主）整理、提炼而成，用于系统学习 Python 数据可视化库 Matplotlib。
> 整理日期：2026-10-13　|　对应版本：Matplotlib 3.11.x（官方 stable 文档）

---

## 目录

1. [Matplotlib 是什么](#1-matplotlib-是什么)
2. [安装与环境](#2-安装与环境)
3. [核心概念：Figure / Axes / Axis / Artist](#3-核心概念figure--axes--axis--artist)
4. [两种编程风格：OO 与 pyplot](#4-两种编程风格oo-与-pyplot)
5. [最小可用示例](#5-最小可用示例)
6. [绘图输入数据类型](#6-绘图输入数据类型)
7. [常用图表类型速查](#7-常用图表类型速查)
8. [样式定制：颜色、线型、标记](#8-样式定制)
9. [子图与布局管理](#9-子图与布局管理)
10. [标题、标签、图例与文本标注](#10-标题标签图例与文本标注)
11. [坐标轴控制](#11-坐标轴控制)
12. [配色方案与色彩映射（colormap）](#12-配色方案与色彩映射colormap)
13. [全局配置 rcParams 与样式表](#13-全局配置-rcparams-与样式表)
14. [保存图片与后端（backend）](#14-保存图片与后端backend)
15. [动画与交互](#15-动画与交互)
16. [常见问题与避坑清单](#16-常见问题与避坑清单)
17. [推荐学习路径](#17-推荐学习路径)
18. [文本教程资源链接](#18-文本教程资源链接)
19. [视频教程链接](#19-视频教程链接)

---

## 1. Matplotlib 是什么

Matplotlib 是 Python 生态中最基础、最成熟的**二维（及简单三维）绘图库**，用于把数据画成折线图、散点图、柱状图、直方图、热力图等，是 NumPy / Pandas 数据分析流程的事实标准可视化工具。

官方给出的定位：把数据画在 `Figure`（画布）上，一个 Figure 可以包含一个或多个 `Axes`（坐标系/绘图区），数据点在 Axes 中以 x-y 坐标（极坐标下 θ-r，3D 下 x-y-z）表示。

主要特点：

- **面向对象 + 状态机双接口**：既支持 `plt.xxx()` 快速画图，也支持 `fig, ax = plt.subplots()` 的显式对象风格。
- **高度可定制**：几乎图上的每个元素（线宽、刻度、字体、图例、颜色映射）都能精细控制。
- **可发布质量**：支持导出 PNG / JPG / SVG / PDF / EPS 等格式，可直接用于论文与报告。
- **可扩展**：内置 `mplot3d`、动画、图像处理、极坐标、地理绘图等工具包。

---

## 2. 安装与环境

```bash
# 基础安装
pip install matplotlib

# 国内加速
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

# 常用搭档（教程几乎都要求）
pip install numpy pandas

# 使用 conda
conda install matplotlib
```

验证安装：

```python
import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.__version__)
```

环境说明：

- **Jupyter Notebook / JupyterLab / Colab**：图形会自动显示，通常**不需要**写 `plt.show()`。
- **普通 Python 脚本 / IDE**：需要显式调用 `plt.show()` 才会弹出窗口。
- **无图形界面的服务器**：使用 `Agg` 后端（非交互式），配合 `savefig()` 保存图片即可。
- 中文显示问题见[第 16 节](#16-常见问题与避坑清单)。

---

## 3. 核心概念

Matplotlib 的图形由一套层次化的对象组成，理解这四个概念是看懂官方文档的关键。

| 对象 | 含义 |
| --- | --- |
| **Figure** | 整张画布/图，可理解为"一张纸"。它管理所有子级 Axes、特殊 Artist（标题、图例、色条）以及嵌套子图。 |
| **Axes** | 一个绘图区（坐标系），是**绝大多数绘图方法的入口**，通常包含两个（3D 时为三个）Axis 对象，并有标题、x/y 标签。注意与 Axis 区分：Axes 是"坐标系"，Axis 是"坐标轴"。 |
| **Axis** | 坐标轴对象，负责设置刻度范围并生成刻度（tick）与刻度标签（tick label）。刻度位置由 `Locator` 决定，标签格式由 `Formatter` 决定。 |
| **Artist** | 图上**一切可见元素**都是 Artist，包括 Figure、Axes、Axis、`Text`、`Line2D`、`Collection`（散点）、`Patch`（矩形/多边形）等。渲染时所有 Artist 被绘制到 canvas 上。 |

创建 Figure 的常见方式：

```python
fig = plt.figure()                    # 空白 figure，无 Axes
fig, ax = plt.subplots()              # 单 Axes
fig, axs = plt.subplots(2, 2)         # 2x2 网格 Axes
fig, axs = plt.subplot_mosaic([['left', 'right_top'],
                               ['left', 'right_bottom']])  # 复杂非均匀布局
```

> 官方建议：复杂图、可复用函数与脚本中，优先使用**显式（OO）风格**；只在快速交互式探索时用 pyplot 风格。`pylab`（`from pylab import *`）已被强烈弃用，不要用。

---

## 4. 两种编程风格

### 4.1 显式/面向对象风格（推荐）

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(x, x,     label='linear')
ax.plot(x, x**2,  label='quadratic')
ax.plot(x, x**3,  label='cubic')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title('Simple Plot')
ax.legend()
```

优点：图与对象一一对应，多子图、多图循环时不串图，函数封装友好。

### 4.2 隐式/pyplot 风格（快速画图）

```python
x = np.linspace(0, 2, 100)

plt.figure(figsize=(5, 2.7), layout='constrained')
plt.plot(x, x,    label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Simple Plot')
plt.legend()
```

pyplot 会自动创建并管理当前 Figure/Axes（"当前图"概念），写起来短，但图多了容易混乱。

### 4.3 可复用的封装函数（官方推荐签名）

```python
def my_plotter(ax, data1, data2, param_dict):
    """A helper function to make a graph."""
    out = ax.plot(data1, data2, **param_dict)
    return out

data1, data2, data3, data4 = np.random.randn(4, 100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})
```

要点：**把 `ax` 作为第一个参数传入**，避免函数内部依赖 pyplot 全局状态。

---

## 5. 最小可用示例

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()             # 创建包含单个 Axes 的 Figure
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # 在 Axes 上画数据
plt.show()                           # 显示图形（Jupyter 中可省略）
```

这就是官方快速入门的第一段代码，所有复杂图形都是它的扩展。

---

## 6. 绘图输入数据类型

- 绘图函数期望输入 `numpy.array` 或 `numpy.ma.masked_array`，以及任何可被 `np.asarray()` 转换的对象。
- 类似数组的"array-like"（如 `pandas` 对象、`numpy.matrix`）**可能不会被正确处理**，推荐先转换：

```python
b = np.matrix([[1, 2], [3, 4]])
b_asarray = np.asarray(b)
```

- 大多数方法支持传入可字符串索引的对象（dict、结构化数组、`pandas.DataFrame`），并用 `data=` 关键字引用列名：

```python
np.random.seed(19680801)
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.scatter('a', 'b', c='c', s='d', data=data)   # 直接写列名，配合 data=
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')
```

这种写法在配合 Pandas DataFrame 时非常方便。

---

## 7. 常用图表类型速查

以下示例统一采用 `fig, ax = plt.subplots()` 风格。

### 7.1 折线图 `plot`

```python
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.legend()
```

### 7.2 散点图 `scatter`

```python
x = np.random.randn(200)
y = np.random.randn(200)
colors = np.random.rand(200)
sizes = 200 * np.random.rand(200)

ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis',
           edgecolor='k', linewidths=0.5)
fig.colorbar(ax.collections[0], ax=ax, label='value')  # 添加色条
```

`c` 控制颜色（可映射到 colormap），`s` 控制点面积，`alpha` 控制透明度。

### 7.3 柱状图 / 条形图 `bar` / `barh`

```python
labels = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 9]

ax.bar(labels, values, color='steelblue', width=0.6)
ax.bar_label(ax.containers[0])   # 自动在柱顶加数值标签（Matplotlib ≥3.4）
```

分组柱状图：把 x 位置错开 `width` 即可；堆叠柱状图使用 `bottom=` 参数。

### 7.4 直方图 `hist`

```python
data = np.random.randn(1000)
ax.hist(data, bins=30, density=True, alpha=0.7, edgecolor='white')

# 二维直方图
# ax.hist2d(x, y, bins=30, cmap='Blues')
```

### 7.5 饼图 `pie`

```python
sizes = [30, 20, 25, 25]
ax.pie(sizes, labels=['A', 'B', 'C', 'D'],
       autopct='%1.1f%%', startangle=90, explode=(0.05, 0, 0, 0))
ax.axis('equal')   # 保证是正圆
```

### 7.6 箱线图 `boxplot` / 小提琴图 `violinplot`

```python
data = [np.random.normal(0, 1, 100) for _ in range(4)]
ax.boxplot(data, tick_labels=['A', 'B', 'C', 'D'], patch_artist=True)

# ax.violinplot(data, showmeans=True)
```

### 7.7 误差棒 `errorbar`

```python
x = np.arange(5)
y = [1, 3, 2, 5, 4]
yerr = [0.3, 0.4, 0.2, 0.5, 0.3]

ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5)
```

### 7.8 填充区域 `fill_between`

```python
x = np.linspace(0, 10, 200)
y = np.sin(x)
ax.plot(x, y)
ax.fill_between(x, y - 0.2, y + 0.2, alpha=0.3, label='置信区间')
ax.legend()
```

### 7.9 热力图 / 图像 `imshow`

```python
matrix = np.random.rand(10, 10)
im = ax.imshow(matrix, cmap='hot', interpolation='nearest',
               aspect='auto', origin='lower')
fig.colorbar(im, ax=ax)
```

`imshow` 对数组默认以左上角为原点，可用 `origin='lower'` 改为左下角。

### 7.10 等高线图 `contour` / `contourf`

```python
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = np.exp(-(X**2 + Y**2))

cs = ax.contour(X, Y, Z, levels=10, cmap='viridis')
ax.clabel(cs, inline=True, fontsize=8)   # 标注等高线数值
# ax.contourf(X, Y, Z, levels=20, cmap='viridis')  # 填充版
```

### 7.11 双 y 轴 `twinx`

```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, np.sin(x), 'g-')
ax2.plot(x, np.cos(x) * 10, 'b--')
ax1.set_ylabel('sin', color='g')
ax2.set_ylabel('cos*10', color='b')
```

### 7.12 极坐标图

```python
theta = np.linspace(0, 2 * np.pi, 100)
ax.remove()
ax = fig.add_subplot(projection='polar')
ax.plot(theta, np.abs(np.sin(3 * theta)))
```

### 7.13 3D 图（mplot3d）

```python
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
ax.scatter([0], [0], [1], color='red', s=50)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
```

其他常用：`stackplot`（堆叠面积）、`stem`（火柴杆）、`step`（阶梯）、`hexbin`（六边形分箱）、`pcolormesh`（伪彩图）、`quiver`/`streamplot`（向量场）、`eventplot`、`spy`（稀疏矩阵）。

> 官方 **Plot types** 页面按类别（Relational / Distribution / Categorical / Part-whole / Images / 3D 等）整理了全部图表类型与示例代码，是最好的选型索引。

---

## 8. 样式定制

### 8.1 三种设置方式

1. **调用绘图方法时传入**（一次性）：
   ```python
   ax.plot(x, y, color='blue', linewidth=3, linestyle='--')
   ```
2. **通过 setter 事后修改**（针对已有 Artist）：
   ```python
   line, = ax.plot(x, y, color='orange', linewidth=2)
   line.set_linestyle(':')
   ```
3. **全局 rcParams / 样式表**（影响所有后续图），见第 13 节。

### 8.2 颜色（Color）

支持多种写法：

| 写法 | 示例 |
| --- | --- |
| 单字符简写 | `'r'` `'g'` `'b'` `'k'` `'w'` |
| 颜色名 | `'steelblue'` `'cornflowerblue'` |
| 十六进制 | `'#1f77b4'` |
| RGB(A) 元组 | `(0.1, 0.2, 0.5)` / `(0.1, 0.2, 0.5, 0.8)` |
| 灰度字符串 | `'0.75'` |
| 颜色循环 | `'C0'` `'C1'` … `'C9'`（默认色环第 n 个颜色） |
| 颜色映射值 | 配合 `cmap=` + 数值数组 |

散点可以分别指定内部与边缘颜色：

```python
ax.scatter(data1, data2, s=50, facecolor='C0', edgecolor='k')
```

### 8.3 线宽、线型、标记

- 线宽单位为**排版点**（1 pt = 1/72 英寸）。
- 线型：`'-'` 实线、`'--'` 虚线、`'-.'` 点划线、`':'` 点线；也可用 `(offset, (on, off, ...))` 自定义。
- 标记大小：`plot` 的 `markersize` 单位是点；`scatter` 的 `s` 近似正比于**面积**。
- 常用标记：`'o'` 圆、`'s'` 方、`'^'` `'v'` 三角、`'d'` 菱形、`'x'` 叉、`'*'` 星、`'+'` 十字、`'.'` 点。

```python
ax.plot(data1, 'o', label='data1')
ax.plot(data2, 'd', label='data2')
ax.plot(data3, 'v', label='data3')
ax.plot(data4, 's', label='data4')
ax.legend()
```

标记也可用 `MarkerStyle` 完全自定义，参考官方 *Marker reference*。

### 8.4 其他常用样式参数

| 参数 | 作用 |
| --- | --- |
| `alpha` | 透明度（0–1） |
| `label` | 图例文字 |
| `zorder` | 叠放顺序 |
| `markerfacecolor` / `markeredgecolor` | 标记填充/边缘色 |
| `drawstyle` | `'default'` / `'steps'` 等 |
| `solid_capstyle` | 线端样式 |
| `hatch` / `fill`（柱状） | 填充图案 |

---

## 9. 子图与布局管理

### 9.1 `plt.subplots` 网格布局

```python
fig, axs = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')

axs[0, 0].plot(x, x)
axs[0, 1].scatter(x, x**2)
axs[1, 0].bar(['a', 'b', 'c'], [1, 2, 3])
axs[1, 1].hist(np.random.randn(500), bins=20)

for ax in axs.flat:          # 遍历所有子图统一设置
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)
```

**注意**：`plt.subplots(1, 1)` 返回单个 Axes（不是数组）；`(1, N)` 或 `(N, 1)` 返回一维数组；`(M, N)` 返回二维数组。

### 9.2 `subplot_mosaic` 复杂布局（推荐）

```python
fig, axd = plt.subplot_mosaic(
    [['main', 'main', 'side'],
     ['main', 'main', 'side'],
     ['bottom', 'bottom', 'bottom']],
    figsize=(10, 6), layout='constrained')

axd['main'].plot(x, np.sin(x))
axd['side'].bar(['a', 'b'], [3, 5])
axd['bottom'].hist(np.random.randn(500))
```

用字符串拼图描述布局，比 `GridSpec` 直观得多，返回 dict 可按名字取子图。

### 9.3 `GridSpec` 精细控制

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 3)
ax1 = fig.add_subplot(gs[0, :])     # 跨整行
ax2 = fig.add_subplot(gs[1:, :2])   # 跨多行多列
ax3 = fig.add_subplot(gs[1:, 2])
```

支持 `height_ratios`、`width_ratios`、`hspace`、`wspace` 控制比例与间距。

### 9.4 自动排列算法

- **`layout='constrained'`（推荐）**：在 `plt.subplots()` / `figure()` 时指定，自动防止标签重叠并支持 colorbar 等复杂元素。
- **`fig.tight_layout()`**：老方案，官方轻度不推荐，遇到 colorbar/嵌套布局时容易失效。
- **手动**：`fig.subplots_adjust(left=, right=, top=, bottom=, wspace=, hspace=)`。

### 9.5 嵌套子图

```python
fig = plt.figure(figsize=(8, 5))
outer = fig.add_gridspec(1, 2)
ax_left = fig.add_subplot(outer[0])
inner = outer[1].subgridspec(2, 1)
ax_top = fig.add_subplot(inner[0])
ax_bot = fig.add_subplot(inner[1])
```

---

## 10. 标题、标签、图例与文本标注

### 10.1 基本文字

```python
ax.set_title('Title', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)

# 图上任意位置写字，坐标是数据坐标
ax.text(2.5, 0.5, 'peak', fontsize=11, color='red', ha='center', va='bottom')
```

### 10.2 图例 `legend`

```python
ax.plot(x, y1, label='数据 A')   # 必须有 label
ax.plot(x, y2, label='数据 B')

ax.legend(loc='upper right',          # 位置：best/upper left/center/lower right...
          ncol=2,                     # 列数
          frameon=True,
          fontsize=10,
          title='图例标题',
          shadow=True)
```

其他方式：`ax.legend(handles, labels)` 手动传入；`fig.legend()` 做整图级图例；`ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')` 把图例放到坐标区外。

### 10.3 箭头标注 `annotate`

```python
ax.annotate('极小值',
            xy=(0, -1),                    # 被标注点的位置
            xytext=(2, -0.5),              # 文字位置
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))
```

`arrowprops` 支持 `width`/`headwidth` 旧式参数与 `arrowstyle` 新式参数两种写法。

### 10.4 数学公式（Mathtext）

用 `$...$` 包裹，无需 LaTeX 环境：

```python
ax.set_title(r'$\alpha > \beta$, $\sum_{i=0}^{N} x_i^2$')
ax.set_xlabel(r'Time $t$ (s)')
```

需要更强排版时可用 `usetex=True` 调用真实 LaTeX（需本地安装 TeX）。

---

## 11. 坐标轴控制

```python
# 范围与比例
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xscale('log')        # 'linear' / 'log' / 'symlog' / 'logit'
ax.invert_yaxis()           # 反转 y 轴

# 刻度位置与标签
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['0', '二', '四', '六', '八', '十'], rotation=45, ha='right')
ax.tick_params(axis='both', which='major', labelsize=10, length=6)
ax.minorticks_on()          # 打开次刻度

# 网格
ax.grid(True, which='major', linestyle='--', alpha=0.5)

# 边框脊线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 开关坐标轴
ax.axis('off')      # 隐藏
ax.axis('equal')    # 等比例（饼图/圆必备）
```

高级刻度控制使用 `matplotlib.ticker`：

```python
from matplotlib.ticker import MultipleLocator, FuncFormatter, PercentFormatter

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.1f}%'))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
```

**日期轴**：

```python
import matplotlib.dates as mdates

ax.plot(dates, values)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()      # 自动旋转日期标签
```

**双轴 / 次坐标轴**：见 7.11 `twinx()` / `twiny()`。

---

## 12. 配色方案与色彩映射

- **典型连续 colormap**：`viridis`（默认，色盲友好）、`plasma`、`inferno`、`magma`、`cividis`、`Blues`、`Reds`。
- **发散型**：`coolwarm`、`RdBu`、`bwr`、`seismic`（适合有中心零点的数据）。
- **顺序型/循环型**：`twilight`、`hsv`、`tab10`（分类色）。
- **定性色板**：`tab10`、`Set1`、`Paired`、`Dark2`。

```python
sc = ax.scatter(x, y, c=values, cmap='viridis')
fig.colorbar(sc, ax=ax, label='数值', orientation='vertical')

# 归一化：把数据压到 0-1 之外也可以
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
norm = LogNorm(vmin=1, vmax=1000)
ax.imshow(data, norm=norm, cmap='viridis')
```

**色条（colorbar）常用参数**：`shrink`（长度比例）、`pad`（与主图间距）、`aspect`、`extend='both'`（显示超出范围箭头）。

**选色建议（官方 *Choosing Colormaps*）**：

- 连续型数据 → 用**感知均匀**的 colormap（如 viridis），**不要用** 老式的 `jet`/`rainbow`（亮度非单调，会制造虚假结构）。
- 有明确中点 → 用发散型并设置 `TwoSlopeNorm`/`centered`。
- 分类数据 → 用定性色板，不要把连续 colormap 用在类别上。

---

## 13. 全局配置 rcParams 与样式表

### 13.1 rcParams

```python
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (8, 5)
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.unicode_minus'] = False      # 解决负号显示为方块
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 全局中文黑体
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['savefig.bbox'] = 'tight'
```

也可用上下文管理器临时生效：

```python
with mpl.rc_context({'lines.linewidth': 3, 'axes.grid': True}):
    fig, ax = plt.subplots()
    ax.plot(x, y)
```

### 13.2 样式表 `style`

```python
print(plt.style.available)   # 查看所有内置样式
plt.style.use('ggplot')      # 整体切换外观

# 组合多个样式
plt.style.use(['dark_background', 'presentation'])

# 临时使用
with plt.style.context('seaborn-v0_8-whitegrid'):
    fig, ax = plt.subplots()
    ax.plot(x, y)
```

常见内置样式：`ggplot`、`seaborn-v0_8-*`、`bmh`、`dark_background`、`classic`、`fivethirtyeight`、`grayscale`。

### 13.3 保存/加载自定义样式

```python
mpl.rc_file('my_style.mplstyle')
```

---

## 14. 保存图片与后端

### 14.1 保存

```python
fig.savefig('plot.png', dpi=300, bbox_inches='tight', transparent=False)
fig.savefig('plot.pdf')    # 矢量图，适合论文
fig.savefig('plot.svg')    # 矢量图，适合网页
```

要点：

- `dpi` 控制分辨率，论文插图一般 300 以上。
- `bbox_inches='tight'` 去掉多余白边。
- **务必在 `plt.show()` 之前保存**，因为某些后端在 `show()` 后 figure 会被清空。
- 支持 `png/jpg/tiff/pdf/svg/eps/ps/webp` 等。

### 14.2 后端（Backend）

后端负责把 Figure 渲染到屏幕文件或 GUI 窗口：

- **交互式**：`TkAgg`（默认常见）、`QtAgg`、`GTK4Agg`、`MacOSX`、`nbagg`/`ipympl`（Notebook）。
- **非交互式**：`Agg`（位图）、`pdf`、`svg`、`ps`、`cairo`、`pgf`。

```python
import matplotlib
matplotlib.use('Agg')     # 必须在 pyplot 导入之前调用
import matplotlib.pyplot as plt
```

在 Jupyter 中使用交互式后端：

```python
%matplotlib inline      # 静态内嵌（默认）
%matplotlib widget      # 需要 ipympl，支持缩放/平移
%matplotlib notebook    # 老式交互
```

---

## 15. 动画与交互

### 15.1 `FuncAnimation`

```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 200)
line, = ax.plot(x, np.sin(x))

def update(frame):
    line.set_ydata(np.sin(x + frame / 10))
    return line,

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
ani.save('wave.gif', writer='pillow', fps=20)   # 或 writer='ffmpeg'
plt.show()
```

- `blit=True` 只重绘变化部分，大幅提速（见官方 *Faster rendering by using blitting*）。
- **必须保留对动画对象的引用**（如 `ani = ...`），否则会被垃圾回收导致不动。
- 保存 GIF 需要 `pillow`，保存 MP4 需要 `ffmpeg`。

### 15.2 交互

- 缩放/平移：工具栏按钮，或 `ax.set_navigate(True)`。
- 键盘/鼠标事件：`fig.canvas.mpl_connect('button_press_event', on_click)`。
- 交互式组件：`Button`、`Slider`、`RadioButtons`、`CheckButtons`、`TextBox`（`matplotlib.widgets`）。
- `mplcursors`、`ipympl`、`plotly` 等第三方库可获得更现代的交互体验。

---

## 16. 常见问题与避坑清单

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| **中文显示成方块（豆腐块）** | 默认字体不含中文字形 | `plt.rcParams['font.sans-serif'] = ['SimHei']`（Windows）/ `['Heiti TC']`（macOS）/ `['WenQuanYi Micro Hei']`（Linux）；并设 `plt.rcParams['axes.unicode_minus'] = False` |
| 负号显示为方块 | 同上，Unicode 负号缺失 | `axes.unicode_minus = False` |
| 图像不显示 | 在脚本中没调用 `plt.show()`，或后端为 Agg | 加 `plt.show()`；Jupyter 用 `%matplotlib inline` |
| `plt.show()` 后保存是空白 | show 之后 figure 状态变化 | 先 `savefig()` 再 `show()` |
| 图例不显示 | 绘图时没写 `label=` | 补 `label=`，或显式传 `ax.legend(handles, labels)` |
| 子图标签被裁切 | 布局算法未生效 | 用 `layout='constrained'`，或 `bbox_inches='tight'` 保存 |
| 内存持续增长（循环画图） | 每次循环新建 figure 未关闭 | 循环内 `plt.close(fig)`；大批量绘图用 OO 风格复用 figure |
| 保存大图很慢 | dpi 过高、元素过多 | 降低 dpi，散点用 `rasterized=True`，大数据改用 `pcolormesh`/`hist2d` |
| `Axes` 与 `Axis` 混淆 | 命名相近 | Axes = 坐标系（画图入口），Axis = 坐标轴（刻度） |
| 用 `pylab` | 已弃用 | 改用 `import matplotlib.pyplot as plt` |
| 3D 图默认视角差 | 默认仰角/方位角不合适 | `ax.view_init(elev=25, azim=-60)` |
| 色条与主图重叠 | 未使用布局算法 | `layout='constrained'` 或在 `colorbar()` 中设 `pad`/`fraction` |

**快速设置中文与美观默认值（推荐放在脚本开头）**：

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.figsize': (8, 5),
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})
```

---

## 17. 推荐学习路径

**第 1 阶段：入门（1–2 天）**
1. 安装并跑通官方 Quick start guide；
2. 理解 Figure / Axes / Artist 层次；
3. 掌握 `subplots` + `plot` + `set_xlabel/set_ylabel/set_title/legend` 五件套；
4. 用 `savefig` 导出图片。

**第 2 阶段：常用图形（2–3 天）**
5. 逐个练习 scatter / bar / hist / pie / boxplot / errorbar / fill_between / imshow / contour；
6. 学会 `subplots` 网格与 `subplot_mosaic` 布局；
7. 学会 `twinx`、colorbar、日期轴、对数轴。

**第 3 阶段：进阶（1 周）**
8. 样式表与 rcParams，做出一套统一风格；
9. 认真读官方 *Lifecycle of a Plot*、*Artist tutorial*、*Customizing Matplotlib*；
10. 颜色映射选择（Choosing Colormaps）；
11. 动画 FuncAnimation + blitting；
12. 3D（mplot3d）与工具箱（axes_grid1、axisartist）。

**第 4 阶段：实战**
13. 用 Pandas 读数据 → 清洗 → 用 Matplotlib 出图 → 导出 PDF 报告；
14. 从官方 Gallery 找相似图，复制代码改成自己的数据（最快的成长方式）；
15. 用 Python Graph Gallery 参考配色与图表选型。

**官方文档结构导航（重要）**

| 板块 | 内容 | 适合阶段 |
| --- | --- | --- |
| Quick start guide | 基本模式与最佳实践 | 入门 |
| Tutorials（Pyplot / Images / Lifecycle / Artist） | 系统教程 | 入门—进阶 |
| Plot types | 按类别索引所有图表类型 | 全程 |
| Examples（Gallery） | 上千个可运行示例 | 全程 |
| User guide（Axes、Artists、Colors、Text、Animations、Toolkits） | 深入原理 | 进阶 |
| API Reference | 查参数 | 全程 |
| Cheatsheets（官方速查表） | 1 页 A4 打印速查 | 复习 |

官方还提供 **Beginner / Intermediate / Tips** 三张速查表（PDF + PNG，A4 可打印）：
[matplotlib/cheatsheets](https://github.com/matplotlib/cheatsheets)

---

## 18. 文本教程资源链接

### 18.1 官方（首选）

- [官方首页 / Getting Started](https://matplotlib.org/stable/)
- [Quick start guide（快速入门）](https://matplotlib.org/stable/users/explain/quick_start.html) — 本文档第 3–8 节的主要来源
- [Tutorials 教程索引](https://matplotlib.org/stable/tutorials/index.html) — Pyplot tutorial、Image tutorial、Lifecycle of a Plot、Artist tutorial、Coding shortcuts
- [Using Matplotlib 用户指南](https://matplotlib.org/stable/users/index.html) — Axes、Artists、Colors、Text、Animations、Toolkits、FAQ
- [Plot types 图表类型](https://matplotlib.org/stable/plot_types/index.html)
- [Examples Gallery 示例库](https://matplotlib.org/stable/gallery/index.html)
- [API Reference](https://matplotlib.org/stable/api/index.html)
- [Cheatsheets 速查表仓库](https://github.com/matplotlib/cheatsheets)
- [External resources（官方整理的外部资源）](https://matplotlib.org/stable/users/resources/index.html)

### 18.2 高质量第三方文本教程

- [Matplotlib tutorial — Nicolas P. Rougier](https://www.labri.fr/perso/nrougier/teaching/matplotlib/) — 简洁高效，官方推荐
- [Scientific Visualization: Python + Matplotlib（免费书，2021）](https://hal.inria.fr/hal-03427242/) — 进阶最佳读物
- [Anatomy of Matplotlib（IPython Notebook 教程）](https://github.com/WeatherGod/AnatomyOfMatplotlib)
- [Learning Scientific Programming with Python, Ch.7](https://scipython.com/book/chapter-7-matplotlib/)
- [Python Graph Gallery](https://www.python-graph-gallery.com) — 按图表类型找代码，选型神器
- [Matplotlib Journey 交互式在线课程](https://www.matplotlib-journey.com/)
- [Beyond the Basics: Data Visualization in Python](https://github.com/stefmolin/python-data-viz-workshop)
- [Matplotlib 中文网教程](https://kr.matplotlib.net/tutorial/)
- [osgeo.cn 镜像的中文文档教程](https://osgeo.cn/matplotlib/tutorials/)
- [Ten Simple Rules for Better Figures（论文配图规范）](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)

---

## 19. 视频教程链接

> 说明：以下仅为链接汇总，未做内容总结。中文以 B 站为主，英文以 YouTube 与官方推荐为主。

### 中文（B 站）

| 视频 | 链接 |
| --- | --- |
| 【全 24 集】Matplotlib（绘图）数据可视化教程，由浅入深，零基础入门到精通 | <https://www.bilibili.com/video/BV1P3ej6zEGv/> |
| 【莫烦Python】Matplotlib Python 画图教程 | <https://www.bilibili.com/video/BV1tZf8YQEQB/> |
| 【全 93 集】Python 数据分析（Numpy+Pandas+Matplotlib）数据可视化教程 | <https://www.bilibili.com/video/BV1WiVc6yESq/> |
| 【2026 最新版】Numpy+Pandas+Matplotlib —— Python 数据分析与数据可视化 | <https://www.bilibili.com/video/BV12Hac6fETB/> |
| 黑马程序员 Python 数据分析全套（含 Matplotlib 数据处理与统计） | <https://www.bilibili.com/list/37974444/?oid=115418354222327&bvid=BV1ReshzoEgG> |
| 黑马程序员 数据分析-Matplotlib 基础-图表实战（柱状图） | <https://www.bilibili.com/video/BV1sHU9BmEne/?p=150> |
| 黑马程序员 机器学习入门 day1-15 matplotlib 的基本使用 | <https://www.bilibili.com/video/BV1a7411d7fk/?p=15> |
| 优酷 Matplotlib 专辑列表 | <https://list.youku.com/albumlist/show/id_28097045.html> |

### 英文（YouTube / O'Reilly，均为官方 External resources 推荐）

| 视频 | 链接 |
| --- | --- |
| Plotting with matplotlib — Mike Müller | <https://www.youtube.com/watch?v=P7SVi0YTIuE> |
| Introduction to NumPy and Matplotlib — Eric Jones | <https://www.youtube.com/watch?v=3Fp1zn5ao2M> |
| Anatomy of Matplotlib — Benjamin Root & Hannah Aizenman | <https://www.youtube.com/watch?v=6gdNUDs6QPc> |
| Matplotlib Introduction 播放列表 — codebasics | <https://www.youtube.com/playlist?list=PLeo1K3hjS3uu4Lr8_kro2AqaO6CFYgKOl> |
| Matplotlib — Derek Banas | <https://www.youtube.com/watch?v=wB9C0Mz9gSo> |
| Data Visualization Basics with Python（O'Reilly）— Randal S. Olson | <https://www.oreilly.com/library/view/data-visualization-basics/9781771375573/> |

---

## 附：一页跑通全部核心用法（可复制运行）

```python
"""matplotlib 核心用法速览 —— 直接运行即可出图"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- 全局配置 ----------
mpl.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
})

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 200)

# ---------- 布局：2x2 + 跨行 ----------
fig, axd = plt.subplot_mosaic(
    [['line', 'scatter'],
     ['bar',  'hist']],
    layout='constrained', figsize=(10, 6.5))

# 1) 折线 + 填充
ax = axd['line']
ax.plot(x, np.sin(x), color='C0', lw=2, label='sin(x)')
ax.plot(x, np.cos(x), color='C1', lw=2, ls='--', label='cos(x)')
ax.fill_between(x, np.sin(x) - 0.15, np.sin(x) + 0.15, alpha=0.2, color='C0')
ax.set_title('折线图 + 填充')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(loc='upper right')

# 2) 散点 + 色条
ax = axd['scatter']
sx, sy = rng.normal(size=300), rng.normal(size=300)
sc = ax.scatter(sx, sy, c=sx**2 + sy**2, s=40, cmap='viridis',
                alpha=0.75, edgecolor='k', linewidths=0.3)
fig.colorbar(sc, ax=ax, label='r²')
ax.set_title('散点图 + 色彩映射')

# 3) 柱状图 + 数值标签
ax = axd['bar']
labels = ['一月', '二月', '三月', '四月']
values = [23, 41, 35, 52]
bars = ax.bar(labels, values, color='steelblue', width=0.6, edgecolor='white')
ax.bar_label(bars, fmt='%d', padding=3)
ax.set_title('柱状图')
ax.set_ylabel('销量')

# 4) 直方图
ax = axd['hist']
ax.hist(rng.normal(0, 1, 2000), bins=35, density=True,
        color='mediumseagreen', alpha=0.8, edgecolor='white')
ax.set_title('直方图')
ax.set_xlabel('取值'); ax.set_ylabel('密度')

fig.suptitle('Matplotlib 核心图表速览', fontsize=14, fontweight='bold')
fig.savefig('matplotlib_overview.png', bbox_inches='tight')
plt.show()
```

---

*本文档内容整理自 Matplotlib 官方文档（Quick start guide、Tutorials、External resources 等）与网络公开教程，仅用于学习参考。*
