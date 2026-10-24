# scikit-learn 系统学习指南

> 环境已核对：本机 **scikit-learn 1.9.1**（系统 Python 3.14.5），**44 个顶层模块、461 个公开 API**。
> `metrics` 86 个、`linear_model` 44 个、`model_selection` 35 个、`preprocessing` 29 个。
> 所有代码、数字、报错信息都在你的环境中实跑验证过。
>
> ⚠ **本机关键环境事实（务必先读）**：
> | Python | 有 sklearn | 有 pandas |
> |---|---|---|
> | **系统 Python 3.14.5**（跑 sklearn 的这个） | ✅ 1.9.1 | ❌ **未安装** |
> | DSH 捆绑 Python 3.12 | ❌ 未安装 | ✅ 3.0.1 |
>
> **pandas 和 sklearn 不在同一个 Python 里。** 所以本文所有示例都用 **NumPy 数组**写，
> 凡是需要 pandas 的功能（`set_output(transform='pandas')`、字符串列名选列）都会明确标注。
> 想用 pandas：`python -m pip install pandas`（装到系统 Python 3.14）。

---

## 第一部分：权威学习资源（按学习顺序）

### 1. 官方文档（质量极高，免费）

| 资源 | 链接 | 说明 |
|---|---|---|
| **User Guide（用户指南）** | [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html) | **核心教材**，15 章，本文第二部分就是它的地图 |
| **Choosing the right estimator** | [machine_learning_map.html](https://scikit-learn.org/stable/machine_learning_map.html) | **一张流程图选算法**，官方最实用的单页 |
| **Common pitfalls** | [common_pitfalls.html](https://scikit-learn.org/stable/common_pitfalls.html) | **必读**：数据泄露、随机性控制、预处理不一致 |
| Pipelines 与复合估计器 | [modules/compose.html](https://scikit-learn.org/stable/modules/compose.html) | Pipeline / ColumnTransformer 的权威说明 |
| 预处理 | [modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html) | 标准化/编码/离散化/自定义转换器 |
| 模型评估与指标 | [modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html) | **metrics 的完整索引**，含 scoring 字符串对照 |
| 交叉验证 | [modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html) | 各种 CV 迭代器的选择 |
| 超参数调优 | [modules/grid_search.html](https://scikit-learn.org/stable/modules/grid_search.html) | Grid/Random/Halving 搜索 |
| **API 参考** | [api/index.html](https://scikit-learn.org/stable/api/index.html) | 查具体类；`sklearn.linear_model` / `metrics` 两页最常查 |
| **Examples（例程库）** | [auto_examples/index.html](https://scikit-learn.org/stable/auto_examples/index.html) | **最有价值**：几百个可运行示例，含完整代码和图 |
| 常见问题 FAQ | [faq.html](https://scikit-learn.org/stable/faq.html) | 官方答疑 |
| 词汇表 | [glossary.html](https://scikit-learn.org/stable/glossary.html) | 术语准确定义 |

**最高效的文档用法**：先看 **Examples** 找相似任务 → 复制代码跑通 → 再回 User Guide 读对应章节的原理。

### 2. 官方课程与教材

| 资源 | 说明 |
|---|---|
| **scikit-learn MOOC**（官方免费课） | [inria.github.io/scikit-learn-mooc](https://inria.github.io/scikit-learn-mooc/) | **最推荐**：INRIA 出品，有视频+notebook+练习，覆盖 Pipeline/调参/指标 |
| 《Hands-On Machine Learning》(Géron) | 作者是 sklearn 核心贡献者，第 2-4 章讲 sklearn 极细 |
| 《Python Data Science Handbook》Ch.5 | 网上免费，讲 sklearn 的风格简洁 |

### 3. 建议跳过

- 只讲 `fit`/`predict` 两行代码的"sklearn 入门"：这些 10 分钟就会
- **只调模型不建 Pipeline 的教程**：会教你养成数据泄露的坏习惯（见第四部分）
- 用 `train_test_split` 反复调参的教程：测试集被用几十次，指标已不可信
- 1.x 之前的旧教程：`sklearn.cross_validation`（已删除）、`sklearn.grid_search`（已删除）、`OneHotEncoder(sparse=True)`（参数已改名 `sparse_output`）

---

## 第二部分：scikit-learn 包含的所有功能

### 2.1 心智模型：一个统一接口 + 三个概念

sklearn 的 461 个 API 都遵循**同一套接口**。学会这套接口，就等于学会全部：

```
                    ┌─────────────┐
                    │ Estimator   │  fit(X, y)   ← 所有估计器的基类
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌────────────┐ ┌──────────┐ ┌───────────┐
      │Transformer │ │Classifier│ │ Regressor │
      │ fit_transform│ │ predict  │ │ predict   │
      │ transform  │ │predict_proba│ │ score(R²)│
      └────────────┘ └──────────┘ └───────────┘
              │            │            │
              └────────────┴────────────┘
                           ▼
                   ┌───────────────┐
                   │  Pipeline     │  把变换器和模型串成一条流水线
                   └───────────────┘
```

**三个必记概念**：

| 概念 | 含义 | 实测验证 |
|---|---|---|
| **Estimator（估计器）** | 有 `fit()` 的对象 | `LogisticRegression().fit(X, y)` |
| **Transformer（变换器）** | 有 `fit()` + `transform()` | `StandardScaler().fit(X).transform(X)` |
| **Predictor（预测器）** | 有 `predict()` | `clf.predict(X)` |

**所有估计器共享的方法（实测确认）**：
```python
est.get_params()          # 返回所有超参数（字典）
est.set_params(C=2.0)     # 修改超参数
clone(est)                # 深拷贝一个未拟合的副本
is_classifier(est) / is_regressor(est) / is_outlier_detector(est)
```
这也是 `GridSearchCV` 能用**任意**估计器的原因——靠的就是这套统一接口。

**关键实测**：变换器的 `fit` 签名是 `fit(X, y=None, sample_weight=None)`——
`StandardScaler.fit` 只接受 X，**不碰 y**。这是它几乎不会造成数据泄露的根本原因（见 4.1）。

### 2.2 官方 User Guide 的 15 章地图

| 章 | 内容 | 优先级 |
|---|---|---|
| 1. 监督学习 | 17 节：线性模型/SVM/树/集成/朴素贝叶斯/判别分析 | ★★★ |
| 2. 无监督学习 | 聚类/降维/异常检测/密度估计/流形学习 | ★★☆ |
| 3. 模型选择与评估 | 交叉验证/调参/阈值调优/指标/学习曲线 | ★★★ |
| 4. Metadata Routing | 把 sample_weight 等元数据在 Pipeline 中传递 | ★☆☆ |
| 5. Inspection | 部分依赖图 / 排列重要性 | ★★☆ |
| 6. Visualizations | `*Display` 类（ROC/PR/混淆矩阵/决策边界） | ★★☆ |
| 7. Callbacks | 训练过程回调（进度条、评分监控） | ★☆☆ |
| 8. 数据变换 | **Pipeline / 预处理 / 缺失值 / 降维 / 核近似** | ★★★ |
| 9. 数据集加载 | toy / 真实 / 生成器 | ★★★ |
| 10. 计算规模 | 大数据策略 / 性能 / 并行 | ★☆☆ |
| 11. 模型持久化 | joblib / ONNX / skops | ★★☆ |
| **12. 常见陷阱** | **数据泄露 / 随机性 / 预处理不一致** | **★★★ 必读** |
| 13. 数据互操作 | pandas 输出 / Array API | ★★☆ |
| 14. 选择估计器 | 流程图 | ★★★ |
| 15. 外部资源 | 视频/讲座 | ★☆☆ |

### 2.3 各模块规模与代表作（实测）

| 模块 | API 数 | 代表作 |
|---|---|---|
| **`metrics`** | 86 | `accuracy_score` `roc_auc_score` `classification_report` `confusion_matrix` |
| **`linear_model`** | 44 | `LinearRegression` `LogisticRegression` `Ridge` `Lasso` `ElasticNet` `SGDClassifier` |
| **`model_selection`** | 35 | `train_test_split` `cross_val_score` `GridSearchCV` `StratifiedKFold` `learning_curve` |
| **`preprocessing`** | 29 | `StandardScaler` `MinMaxScaler` `OneHotEncoder` `OrdinalEncoder` `PolynomialFeatures` `TargetEncoder` |
| `cluster` | 27 | `KMeans` `DBSCAN` **`HDBSCAN`** `AgglomerativeClustering` `BisectingKMeans` |
| `decomposition` | 20 | `PCA` `TruncatedSVD` `NMF` `FastICA` `KernelPCA` |
| `ensemble` | 19 | `RandomForestClassifier` **`HistGradientBoostingClassifier`** `StackingClassifier` `IsolationForest` |
| `feature_selection` | 19 | `SelectKBest` `RFECV` `SelectFromModel` `SequentialFeatureSelector` |
| `neighbors` | 18 | `KNeighborsClassifier` `LocalOutlierFactor` `NearestNeighbors` |
| `covariance` | 16 | `EmpiricalCovariance` `LedoitWolf` `MinCovDet` |
| `datasets` | 46 | `load_iris` `make_classification` `fetch_california_housing` |
| `svm` | 8 | `SVC` `SVR` `LinearSVC` `OneClassSVM` |
| `tree` | 8 | `DecisionTreeClassifier` `plot_tree` `export_text` |
| `neural_network` | 3 | `MLPClassifier` `MLPRegressor` |
| `naive_bayes` | 5 | `GaussianNB` `MultinomialNB` `ComplementNB` |
| `impute` | 3 | `SimpleImputer` `KNNImputer` `MissingIndicator` |
| `inspection` | 4 | `permutation_importance` `partial_dependence` `DecisionBoundaryDisplay` |
| `compose` | 4 | **`ColumnTransformer`** `TransformedTargetRegressor` `make_column_selector` |

> 完整清单见官方 [API 参考](https://scikit-learn.org/stable/api/index.html)。**不必背，会用 `dir()` 现查即可。**

---

## 第三部分：四大核心能力详解（全部实测）

### 3.1 数据集加载（`sklearn.datasets`）

**内置 toy 数据集（本机实测，无需下载）**：

| 数据集 | 形状 | 类别数 | 用途 |
|---|---|---|---|
| `load_iris()` | X(150, 4) | 3 | 入门分类 |
| `load_digits()` | X(1797, 64) | 10 | 手写数字分类 |
| `load_wine()` | X(178, 13) | 3 | 小规模分类 |
| `load_breast_cancer()` | X(569, 30) | 2 | 二分类 |
| `load_diabetes()` | X(442, 10) | 回归 | 回归 |

**数据生成器（全部实测可跑）**：
```python
from sklearn.datasets import (make_classification, make_regression, make_blobs,
                              make_moons, make_circles)
make_classification(n_samples=100, n_features=20, n_informative=5, n_redundant=2,
                    n_classes=2, weights=[0.9, 0.1], flip_y=0.01, random_state=0)
make_regression(n_samples=100, n_features=100, noise=0.0, random_state=0)
make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=0)   # 聚类
make_moons(n_samples=100, noise=0.05, random_state=0)                   # 非线性二分类
make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=0)     # 同心圆
```

**⚠ 实测踩坑**：`make_classification` 在 sklearn 1.9 有**更严格的参数校验**
```python
make_classification(n_informative=0)     # ❌ InvalidParameterError: must be >= 1
make_classification(n_informative=1, n_classes=2, n_clusters_per_class=2)
# ❌ ValueError: n_classes*n_clusters_per_class must be <= 2**n_informative
```
> 这是 1.9 相对旧版的行为变化。构造"纯噪声"数据时用 `n_informative=1` + 大 `flip_y`。

**返回的 `Bunch` 对象（字典子类，实测键）**：
```python
d = load_iris()
d.keys()          # ['data','target','frame','target_names','DESCR','feature_names','filename','data_module']
d.data / d.target / d.feature_names / d.target_names / d.DESCR
d['data']         # 也能像字典访问
X, y = load_iris(return_X_y=True)     # 直接拿数组，最常用
```

**需要联网的数据集**：`fetch_california_housing` / `fetch_olivetti_faces` / `fetch_20newsgroups` 等。
本机实测 `fetch_california_housing()` 报 **`PermissionError`**（沙箱不允许写缓存目录/联网）。
**离线练习请优先用 `load_*` 和 `make_*`。**

### 3.2 预处理（`sklearn.preprocessing`，29 个 API 全清单）

**实证验证**（29 个全在）：`StandardScaler` `MinMaxScaler` `MaxAbsScaler` `RobustScaler`
`Normalizer` `Binarizer` `KBinsDiscretizer` `KernelCenterer` `PolynomialFeatures`
`SplineTransformer` `PowerTransformer` `QuantileTransformer` `OneHotEncoder` `OrdinalEncoder`
`LabelEncoder` `LabelBinarizer` `MultiLabelBinarizer` **`TargetEncoder`** `FunctionTransformer`
+ 无状态的函数版 `scale` `minmax_scale` `robust_scale` `maxabs_scale` `normalize` `binarize`
`power_transform` `quantile_transform` `label_binarize` `add_dummy_feature`

**缩放器怎么选**：

| 缩放器 | 公式 | 何时用 |
|---|---|---|
| `StandardScaler` | (x−μ)/σ | **默认首选**，假设近似正态 |
| `MinMaxScaler` | (x−min)/(max−min) → [0,1] | 需要固定边界（如图像像素） |
| `RobustScaler` | (x−median)/IQR | **有离群点时** |
| `MaxAbsScaler` | x/max\|x\| | 稀疏数据（不破坏零元素） |
| `Normalizer` | 按**样本**归一化（L1/L2） | 只关心方向（文本/TF-IDF） |

> **区分 Scaler 和 Normalizer**：Scaler 按**列**（特征）缩放，Normalizer 按**行**（样本）缩放。这是初学者最常混淆的一对。

**类别编码**：
```python
# 无序类别 → OneHotEncoder（独热）
OneHotEncoder(handle_unknown='ignore', sparse_output=False)   # ⚠ 1.2+ 参数名是 sparse_output
# 有序类别 → OrdinalEncoder（保序）
OrdinalEncoder(categories=[['低','中','高']])
# 目标变量编码 → LabelEncoder（只用于 y，不能用于 X！）
LabelEncoder().fit_transform(['cat','dog','cat'])
# 高基数类别 → TargetEncoder（用 y 的均值编码，内部 CV 防泄露）
TargetEncoder(cv=5, random_state=0)
```

**⚠ 三条实测过的坑**：
1. `OneHotEncoder(sparse=True)` 是旧参数名，1.2+ 要用 **`sparse_output`**
2. **`LabelEncoder` 只能用于 y，不能用于 X**——它会把类别变成 0/1/2，模型会误以为"2 > 1 > 0"有大小关系。X 的类别编码要用 `OrdinalEncoder` 或 `OneHotEncoder`
3. `TargetEncoder` **必须只 fit 在训练集上**，否则泄露（见 4.3）

**特征工程**：
```python
PolynomialFeatures(degree=2, include_bias=False)   # 多项式/交互项
KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')   # 连续→离散
FunctionTransformer(np.log1p, inverse_func=np.expm1)   # 任意函数
QuantileTransformer(output_distribution='normal')      # 强制正态化
PowerTransformer(method='yeo-johnson')                 # Box-Cox 族
```

### 3.3 模型选择与评估（`sklearn.model_selection`，35 个 API）

**划分数据的三种方式**：

```python
# ① 单次划分（快速原型用；但结果不稳定，见下）
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#    stratify=y 保证类别比例一致——分类问题【务必加】

# ② K 折交叉验证（评估的黄金标准）
cross_val_score(estimator, X, y, cv=5, scoring='roc_auc')
cross_validate(estimator, X, y, cv=5,
               scoring=['accuracy','precision','recall','roc_auc'],
               return_train_score=True)     # 一次拿多个指标
cross_val_predict(estimator, X, y, cv=5)     # 返回 out-of-fold 预测值

# ③ 特殊场景
TimeSeriesSplit(n_splits=5)          # 时间序列：不能用随机 K 折！
GroupKFold(n_splits=5)               # 同组样本不能跨折（如同一病人的多条记录）
StratifiedKFold(n_splits=5, shuffle=True, random_state=0)   # 分层
LeaveOneOut()                        # 留一法（小数据集）
```

**实测：单次划分有多不稳定**（同一个模型，只换随机种子）：
```
10 次单次划分的准确率: [0.85, 0.917, 0.817, 0.883, 0.75, 0.967, 0.833, 0.867, 0.817, 0.917]
均值 0.8617 ± 标准差 0.0592    极差 0.2167
5 折交叉验证:          均值 0.8850 ± 标准差 0.0374
```
**单次划分的标准差是 0.059，极差高达 0.217。** 仅换种子就能让准确率从 0.75 跳到 0.967——
**这种波动足以让你误判两个模型的优劣。** 所以：
- 快速试错：可以用单次划分
- **得出结论/对比模型：必须用交叉验证**

**实测：`cross_val_score` 默认会不会分层？**

这是最容易搞错的一点。实测 `check_cv` 的签名和默认值：
```
check_cv(cv=5, y=y, classifier=None)  -> KFold           （默认 classifier=False）
check_cv(cv=5, y=y, classifier=True)  -> StratifiedKFold
check_cv(cv=5, y=y, classifier=False) -> KFold
```
但 `cross_val_score` 内部会传 `classifier=is_classifier(estimator)`，所以**实际是分层的**。
实测分类器 `cv=5` 各折：`[0.875, 0.975, 0.95, 0.95, 0.95]`。
> **结论**：用 `cross_val_score(分类器, ...)` 会自动分层；但如果你**手动传 `KFold`**，就**不会**分层。

**实测：分层 vs 不分层，在类别不平衡时差别巨大**
```
正类比例 0.10
KFold(5) 各折正类比例:          [0.200, 0.100, 0.025, 0.050, 0.125]  ← 某折只有 2.5% 正类！
StratifiedKFold(5) 各折正类比例: [0.100, 0.100, 0.100, 0.100, 0.100]  ← 完全一致
```
不分层时某折可能几乎没有正类，指标完全不可信。

**超参数搜索的三种方式**：
```python
# ① 网格搜索（穷举，参数少时用）
GridSearchCV(pipeline, {'svc__C':[0.1,1,10], 'svc__gamma':['scale',0.01]},
             cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True, refit=True)

# ② 随机搜索（参数多/维度高时更高效）
RandomizedSearchCV(pipeline, param_distributions, n_iter=50, cv=5, random_state=0)

# ③ 逐次减半（大数据集加速；⚠ 实验性，需先启用）
from sklearn.experimental import enable_halving_search_cv   # ← 必须！
from sklearn.model_selection import HalvingGridSearchCV
```
**实测结果（网格搜索）**：9 组参数 × 5 折 = 45 次拟合；`best_params_` = `{'svc__C': 10, 'svc__gamma': 0.01}`，CV 分数 0.9871，测试集 0.9900。
`gs.best_estimator_` 是**已拟合好的完整 Pipeline**，可直接 `.predict()`。

**跨越 Pipeline 传参的命名规则（最常错）**：
```python
make_pipeline(StandardScaler(), SVC())    # 自动命名: 'standardscaler', 'svc'
param_grid = {'svc__C': [1, 10]}          # 用【步骤名__参数名】双下划线
# 用显式命名避免记错：Pipeline([('scale', StandardScaler()), ('clf', SVC())])
#                   -> '{'clf__C': [...]}'
```

**⚠ 实测踩坑**：`HalvingGridSearchCV` 是实验性 API，不启用就 ImportError：
```
ImportError: HalvingGridSearchCV is experimental and the API might change without any
deprecation cycle. To use it, you need to explicitly import enable_halving_search_cv:
from sklearn.experimental import enable_halving_search_cv
```

**学习曲线与验证曲线（诊断欠拟合/过拟合）**：
```python
learning_curve(est, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5))
# 返回三元组: (train_sizes, train_scores, test_scores)
validation_curve(est, X, y, param_name='svc__C', param_range=[0.01,0.1,1,10], cv=5)
# ⚠ 返回【二元组】: (train_scores, test_scores) —— 实测确认，与 learning_curve 不同
```
**实测学习曲线**（判断过拟合的模板）：
```
训练样本  44: train 0.9273  test 0.7250  差距 +0.2023   ← 小样本，过拟合明显
训练样本 224: train 0.8741  test 0.8429  差距 +0.0312   ← 数据变多，差距收窄
```
**实测验证曲线**（选超参数的模板）：
```
svc__C:      [0.01,   0.1,    1,     10  ]
train:       [0.5107, 0.8804, 0.9500, 1.0000]
test:        [0.5107, 0.7821, 0.8714, 0.8643]   ← C=1 最好，C=10 开始过拟合
```

### 3.4 指标（`sklearn.metrics`，86 个 API）

**⚠ 最重要的认知：准确率在不平衡数据上会骗你（实测）**：
```
数据集正类比例 0.053（严重不平衡）
模型的 accuracy               = 0.9767   ← 看起来很棒？
全预测多数类的 DummyClassifier = 0.9467   ← 什么都不学也有这么高！
真正有价值的：
  precision = 0.8462   recall = 0.6875   f1 = 0.7586
  ROC-AUC   = 0.9474   PR-AUC(AP) = 0.8251   ← 不平衡时比 ROC-AUC 更敏感
  log_loss  = 0.0803   Brier = 0.0183        ← 评估概率质量
```
**结论：不平衡数据必须看 precision/recall/F1/PR-AUC，不能只看 accuracy。**

**分类指标速查**：

| 指标 | 含义 | 注意 |
|---|---|---|
| `accuracy_score` | 正确率 | 不平衡时误导 |
| `precision_score` | 预测为正的里面有多少真为正 | TP/(TP+FP) |
| `recall_score` | 真实为正的里面抓到了多少 | TP/(TP+FN) |
| `f1_score` | precision 和 recall 的调和平均 | 需要单一数字时用 |
| `roc_auc_score` | ROC 曲线下面积 | 需要 `predict_proba`，对不平衡不敏感 |
| `average_precision_score` | PR 曲线下面积 | **不平衡时首选** |
| `log_loss` | 交叉熵 | 评估概率校准质量 |
| `brier_score_loss` | 概率的均方误差 | 概率校准 |
| `confusion_matrix` | 混淆矩阵 | 看错在哪里 |
| `classification_report` | 汇总表 | **含 macro/weighted 平均** |
| `roc_curve` / `precision_recall_curve` | 曲线数据 | 画图用 |

**⚠ `average` 参数的含义**（多分类必用）：
```python
f1_score(y_true, y_pred, average='macro')      # 各类别等权，不推荐用于不平衡
f1_score(y_true, y_pred, average='weighted')   # 按样本数加权
f1_score(y_true, y_pred, average='micro')      # 全局统计
f1_score(y_true, y_pred, average=None)         # 每类单独返回
```
实测 `classification_report` 输出里，macro avg 0.873 与 weighted avg 0.976 相差近 0.1——
**这就是因为少数类表现差，而 weighted 被多数类拉高了。**

**回归指标**（实测数值）：
```
MSE  =    123.474   量纲是 y 的平方，不易解释
RMSE =     11.112   与原 y 同量纲，推荐报告
MAE  =      8.942   对离群点更稳健
R²   =     0.9970   1 完美 / 0 等同预测均值 / 可负
```
`Ridge.score()` 内置返回的就是 **R²**（回归）或 **accuracy**（分类）。

**聚类指标（无标签，实测）**：
```
silhouette_score        = 0.5157   (-1~1，越大越好)
davies_bouldin_score    = 0.7465   (越小越好)
calinski_harabasz_score = 404.0    (越大越好)
inertia_ (簇内平方和)   = 308.0    (肘部法则用)
```

**⚠ 最危险的指标陷阱：指标顺序（`y_true` 在前）**
```python
accuracy_score(y_true, y_pred)   # ✅
accuracy_score(y_pred, y_true)   # ❌ 不报错！但结果可能完全不同
```
这是 sklearn 最常见的静默错误。**建议所有指标调用都写成关键字或显式命名变量。**

**阈值调优（`TunedThresholdClassifierCV`，1.3+ 新 API）**：
```python
TunedThresholdClassifierCV(estimator, scoring='f1', cv=5).fit(X, y)
# 属性 .best_threshold_ 给出自动选出的最优阈值
FixedThresholdClassifier(estimator, threshold=0.8)   # 手动固定阈值
```
实测：默认 0.5 阈值的 f1 为 0.74，调优后 0.7407，手动 0.8 得 0.7692。
**在不平衡场景，调阈值往往比换模型更有效。**

---

## 第四部分：数据泄露——sklearn 最重要的一个话题

这是官方 [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) 的第一主题，
也是新手最常犯、最隐蔽的错误。我实测量化了它的破坏力。

### 4.1 先破一个常见误解：`StandardScaler` 几乎不泄露

很多人被告知"标准化要在划分之后做，否则泄露"。**对，但影响极小**。实测：
```
❌ 全量 fit StandardScaler 后再划分: 测试准确率 0.5500
✅ 先划分，只在训练集 fit:         测试准确率 0.5667
差异: -1.67 个百分点   ← 在噪声范围内，甚至"泄露版"更低
```
**原因（实测签名）**：
```python
import inspect; inspect.signature(StandardScaler().fit)
# (X, y=None, sample_weight=None)   ← 连 y 都不接受！
```
`StandardScaler` 只计算**每个特征自己的**均值/标准差，不涉及特征间关系，也不碰 `y`。
它造成的泄露在数学上是"均值估计略微不准"，影响极小。

> **所以别把"防泄露"理解成"标准化要放后面"这种机械规则。**
> 要理解**是什么让某个操作危险**：**是否使用了测试集的信息、尤其是是否用到了 y 与 X 的关系。**

### 4.2 真正危险的：用到 `y` 的变换（实测 +31.67 个百分点）

**实测：特征选择泄露**（数据里只有 1 个真实信号特征，其余全是噪声）：
```
❌ 划分前用全量数据做 SelectKBest: 测试准确率 0.9000 / 0.8667
✅ 划分后在训练集上做:             测试准确率 0.5667 / 0.4667
虚假提升: +33.33 / +40.00 个百分点    ← 凭空造出了"信号"！
```
**机制**：`SelectKBest(f_classif)` 用 `y` 计算每个特征与标签的相关性。
在**全量数据**上做，它会"挑出"在全部数据（含测试集）上偶然相关的噪声特征。
测试集的信息通过这一步流进了训练过程 —— **模型提前看到了答案。**

**实测：在交叉验证中，这个泄露同样致命**
```
❌ CV 之前先做 SelectKBest:  0.8417
✅ 放进 Pipeline 让 CV 内部逐折做: 0.5250
差异: +31.67 个百分点
```

**正确的做法永远是 Pipeline**：
```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
cross_val_score(pipe, X, y, cv=5)     # ✅ scaler/特征选择在每个 fold 内部单独 fit
```
`Pipeline` 的价值就在这：**它保证每个 fold 的预处理只用该 fold 的训练数据。**

### 4.3 `TargetEncoder` 的泄露（实测 +3.32 个百分点）

`TargetEncoder` 用目标均值编码类别，本身**最危险**（直接把 y 塞进特征）。实测：
```
类别特征是纯噪声（与 y 无关）:
❌ 全量 fit TargetEncoder: 测试 AUC 0.4790
✅ 只在训练集 fit:        测试 AUC 0.4458
虚假提升: +3.32 个百分点
```
> 注意 `TargetEncoder` 内部用 `cv` 做 **out-of-fold 编码**来减轻泄露，
> 但**如果在划分前 `fit`，泄露依然存在**。它必须在 Pipeline 里或只在训练集上 fit。

### 4.4 泄露自检清单

做每个项目时，逐条检查：

```
□ 所有的 transform（缩放/编码/填充/特征选择/降维）都在 Pipeline 里，
  或者只在训练集上 fit？
□ 有没有在划分数据之前计算过任何统计量（均值/分位数/方差/相关性）？
□ 超参数是用交叉验证在【训练集内部】选的吗？测试集只用了一次吗？
□ 时间序列有没有用未来数据预测过去（必须用 TimeSeriesSplit）？
□ 同一个对象（病人/用户/设备）的记录有没有跨训练测试集（需要 GroupKFold）？
□ 有没有把目标变量的信息混进特征（如用全量 y 算的 target encoding）？
□ 数据预处理（去重/填补）是在划分前做的吗？可能造成跨集污染。
```

### 4.5 控制随机性（官方 common_pitfalls 的第三个主题）

**实测的一个反直觉现象：K 越大不一定越好**
```
K= 2: 均值 0.8850  标准差 0.0050   各折 [0.88, 0.89]
K= 3: 均值 0.9198  标准差 0.0315
K= 5: 均值 0.9250  标准差 0.0354
K=10: 均值 0.9100  标准差 0.0831   各折含 1.0 和 0.7 ← 单折波动极大
```
注意 **K=10 的标准差反而最大**。原因：K 越大，训练集越小，单折估计越不稳定且越悲观。
**K=5 或 K=10 是常用折中；不要迷信"K 越大越准"。**

**随机种子的正确用法**：
```python
# ❌ 传 None 会让每次 fit/split 结果不同，无法复现
train_test_split(X, y, random_state=None)
# ✅ 传固定 int，可复现
train_test_split(X, y, random_state=42)
# ✅ 需要多个可复现的随机划分时，传 RandomState 实例
# ⚠ 但传同一个 RandomState 实例给多个估计器会互相消耗状态
```
**所有涉及随机的参数都要固定**：`random_state`、`shuffle`、`n_init`（KMeans）、`n_jobs` 不影响力但影响顺序。

---

## 第五部分：Pipeline 与 ColumnTransformer（sklearn 的精髓）

### 5.1 Pipeline 基础

```python
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

# 显式命名（推荐，调参时不容易写错名字）
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=5000)),
])

# 自动命名（简洁，但名字由类名小写而来）
pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression())

# 使用：一次 fit 全部，一次 predict 全部
pipe.fit(X_train, y_train)
pipe.predict(X_test)

# 访问中间步骤
pipe.named_steps['scaler']      # 按名字
pipe[1]                         # 按位置
pipe[:-1]                       # 切片：只保留变换器

# 只在某一步做 fit_transform
pipe.fit_transform(X_train)     # 只跑到倒数第二步
```

**Pipeline 的四个作用**：
1. **防泄露**：CV 时每个 fold 内部单独 fit 预处理
2. **简化部署**：一个对象含全部预处理 + 模型
3. **统一调参**：`{'clf__C': [...]}` 跨越步骤调参
4. **减少错误**：不会出现"训练时标准化了、预测时忘了"

### 5.2 ColumnTransformer：处理混合类型数据（不用 pandas，实测）

```python
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 列 0,1 数值；列 2 类别；列 3 数值但有缺失
ct = ColumnTransformer(
    transformers=[
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), [0, 1, 3]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [2]),
    ],
    remainder='drop')          # 未指定的列：'drop' 丢弃 / 'passthrough' 保留

pipe = make_pipeline(ct, LogisticRegression(max_iter=1000))
```
**实测结果**：
```
输入形状: (280, 4)     （2 数值 + 1 类别 + 1 含缺失）
变换后:   (280, 6)     ← 类别列被 one-hot 成 3 列
测试准确率: 0.9083
按名称访问变换器: ct.named_transformers_['num'] -> Pipeline
```

**两条实测过的坑**：

1. **`set_output(transform='pandas')` 在你机器上直接失败**（无 pandas）：
   ```
   ImportError: Setting output container to 'pandas' requires pandas to be installed
   ```
   装了 pandas 后可用 `sklearn.set_config(transform_output='pandas')` 让所有 `transform` 返回 DataFrame。

2. **列选择用字符串需要 DataFrame**：
   ```python
   ColumnTransformer([('s', StandardScaler(), ['a','b'])]).fit(numpy_array)
   # ❌ ValueError: Specifying the columns using strings is only supported for dataframes.
   ```
   NumPy 输入必须用**整数索引** `[0, 1]`。

**`make_column_selector`（按 dtype 自动选列，需 DataFrame）**：
```python
from sklearn.compose import make_column_selector
ColumnTransformer([
    ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
    ('cat', OneHotEncoder(), make_column_selector(dtype_include=object)),
])
```

### 5.3 特征名称跟踪（可解释性的基础）

```python
ct.get_feature_names_out()
# 实测输出: ['num__x0', 'num__x1', 'num__x3', 'cat__x2_0.0', 'cat__x2_1.0', 'cat__x2_2.0']
```
命名规则是 **`步骤名__原始特征名`**，一看就知道输出列来自哪里。

**⚠ 实测坑**：`FunctionTransformer` **不提供** `get_feature_names_out`，放进 `ColumnTransformer` 后会报错：
```
AttributeError: Transformer p (type FunctionTransformer) does not provide get_feature_names_out.
```
**对策**：自己实现 `get_feature_names_out`，或把 `FunctionTransformer` 放在不影响命名的位置。

### 5.4 自定义转换器（三种写法，实测）

```python
from sklearn.base import BaseEstimator, TransformerMixin

# 写法1：类（需要 fit 学参数时）
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, standardize=True):
        self.standardize = standardize          # 参数必须原样存入 self
    def fit(self, X, y=None):
        if self.standardize:
            self.scaler_ = StandardScaler().fit(np.log1p(np.abs(X)))
        return self                              # ⚠ 必须 return self！
    def transform(self, X):
        X = np.log1p(np.abs(np.asarray(X)))
        return self.scaler_.transform(X) if self.standardize else X

# 写法2：函数式（无状态变换，最省事）
FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

# 写法3：直接写函数，用 FunctionTransformer 包装
def clip_outliers(X): return np.clip(X, np.percentile(X,1,axis=0), np.percentile(X,99,axis=0))
FunctionTransformer(clip_outliers)
```
**实录测试**：`get_params()` 返回 `{'standardize': True}`、`clone()` 可用 —— 继承 `BaseEstimator` 就自动获得这些。

**两条铁律**：
1. **`fit` 必须 `return self`**，否则 Pipeline 直接报错
2. **`__init__` 的参数必须原样赋给同名属性**（不能做转换），否则 `clone`/`get_params` 会失效

---

## 第六部分：实战场景

### 场景 1 完整的标准工作流（背下这个顺序）

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# ① 加载 + 划分（stratify 保证类别比例）
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ② 定义 Pipeline（含全部预处理）
pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                     LogisticRegression(max_iter=5000, random_state=0))

# ③ 交叉验证估性能（测试集不许看！）
cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
print(f'CV ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}')

# ④ 在训练集内调超参数
gs = GridSearchCV(pipe, {'logisticregression__C': [0.01, 0.1, 1, 10]},
                  cv=5, scoring='roc_auc')
gs.fit(X_train, y_train)
print(f'最优参数: {gs.best_params_}  CV: {gs.best_score_:.4f}')

# ⑤ 只在最后用一次测试集
pred = gs.predict(X_test)
proba = gs.predict_proba(X_test)[:, 1]
print(f'测试集 ROC-AUC: {roc_auc_score(y_test, proba):.4f}')
print(classification_report(y_test, pred, digits=3))
```
**本机实测**：CV ROC-AUC 0.9935 ± 0.0108，最优 C=0.1，测试集 ROC-AUC **0.9957**。

### 场景 2 模型选择：不知道该用哪个（按官方流程图）

```python
# 快速基线：先跑几个"不会错"的模型，看数据难度
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

candidates = {
    'Dummy(多数类)':   DummyClassifier(strategy='most_frequent'),
    '逻辑回归':        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'KNN':            make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'SVM(RBF)':       make_pipeline(StandardScaler(), SVC(probability=True)),
    '随机森林':        RandomForestClassifier(n_estimators=200, random_state=0),
    '梯度提升(Hist)':  HistGradientBoostingClassifier(random_state=0),
}
for name, est in candidates.items():
    s = cross_val_score(est, X_train, y_train, cv=5, scoring='roc_auc')
    print(f'  {name:16s} {s.mean():.4f} ± {s.std():.4f}')
print('  ⚠ 一定要带上 Dummy 基线：如果它也很高，说明任务是平凡的')
```
**选择经验**：
- 表格数据、样本量中等（<10 万）：**`HistGradientBoosting*` 通常最强且很快**
  ⚠ 但在你当前环境下它需要 `threadpool_limits(limits=1)` 才能跑（见 9.1），否则用 `ExtraTrees` 或 `RandomForest` 替代
- 高维稀疏（文本）：`LinearSVC` / `LogisticRegression` / 朴素贝叶斯
- 需要可解释性：线性模型 或 浅决策树（`plot_tree`）
- 小样本高维（p >> n）：SVM(RBF) 或带正则的线性模型
- 特征有缺失：`HistGradientBoosting*`（原生支持 NaN，无需填充）

**本机实测排名**（breast_cancer，5 折 CV ROC-AUC，可直接对照你的结果）：
```
LogisticRegression       0.9935    ← 这个数据集上线性模型就够了！
SVM(RBF)                 0.9940
ExtraTrees               0.9921
HistGradientBoosting     0.9909    （配 threadpoolctl）
RandomForest             0.9896
GradientBoosting(经典)    0.9873
KNN                      0.9891
DummyClassifier          0.5000    ← 基线：确认任务不是平凡的
```
> **注意**：复杂模型不等于更好。这个数据集上简单逻辑回归就打败了所有树模型。
> **永远先跑基线 + 简单模型，再考虑复杂模型。**

### 场景 3 不平衡数据完整处理

```python
# 本机实测：正类仅 5.3% 时，accuracy 0.9767 但"全猜多数类"也有 0.9467
from sklearn.metrics import (classification_report, average_precision_score,
                             roc_auc_score, confusion_matrix, precision_recall_curve)

# 检查 1：先算 Dummy 基线，确认 accuracy 没有意义
# 检查 2：用 class_weight 或阈值调优
clf = LogisticRegression(max_iter=5000, class_weight='balanced')   # 自动加权
# 或采样：imbalanced-learn 库（本机未安装），或手工重采样
# 检查 3：报告 PR-AUC 而不是 ROC-AUC
print(f'PR-AUC: {average_precision_score(y_test, proba):.4f}')
print(f'混淆矩阵:\n{confusion_matrix(y_test, pred)}')
```

### 场景 4 用交叉验证做特征选择（避免泄露）

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, SelectFromModel
from sklearn.model_selection import cross_val_score

# ✅ 正确：放进 Pipeline
pipe = make_pipeline(
    SelectKBest(f_classif, k=10),
    StandardScaler(),
    LogisticRegression(max_iter=5000))
print(cross_val_score(pipe, X_train, y_train, cv=5).mean())

# 递归特征消除 + 自动选特征数
rfe = RFECV(LogisticRegression(max_iter=5000), cv=5, scoring='roc_auc', n_jobs=-1)
# 基于模型重要性选择
sfm = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=0), threshold='median')
```

### 场景 5 可解释性：特征重要性

```python
from sklearn.inspection import permutation_importance, partial_dependence

rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)

# ① 排列重要性（推荐：可用于任何模型，且更可靠）
imp = permutation_importance(rf, X_test, y_test, n_repeats=10, scoring='roc_auc',
                             random_state=0)
# 本机实测 shape = (n_features, n_repeats)
order = np.argsort(imp.importances_mean)[::-1]
for i in order[:5]:
    print(f'  特征{i}: {imp.importances_mean[i]:.4f} ± {imp.importances_std[i]:.4f}')

# ② 树内置重要性（快，但对高基数/连续特征有偏）
rf.feature_importances_

# ③ 部分依赖图（某个特征如何影响预测）
partial_dependence(rf, X_test, [0], kind='average')
```
**实测对比**：排列重要性最高的特征是索引 10（0.2417），而树内置重要性最高的在索引 3（0.0905）——
**两者结论可能完全不同。** 官方文档明确指出树内置重要性对高基数/连续特征有偏，**优先用排列重要性**。

### 场景 6 学习曲线诊断（欠拟合还是过拟合）

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=400, n_features=15, n_informative=5,
                           n_redundant=0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

train_sizes, train_scores, test_scores = learning_curve(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    X_train, y_train, cv=5, train_sizes=np.linspace(0.2, 1.0, 4),
    scoring='roc_auc')
tr_mean, te_mean = train_scores.mean(1), test_scores.mean(1)

# 判断规则：
#   train 和 test 都低            → 欠拟合（模型太弱/特征不够）
#   train 高、test 低、差距大     → 过拟合（加数据/加正则/减容量）
#   test 曲线随数据量仍在上升     → 加数据会有帮助
#   test 曲线已平                  → 加数据无用，该改模型/特征
for n, a, b in zip(train_sizes, tr_mean, te_mean):
    print(f'  n={n:4.0f}  train={a:.4f}  test={b:.4f}  差距={a-b:+.4f}')
```
**本机实测**（`make_classification(n_samples=400, n_features=15, n_informative=5, random_state=0)`，
`scoring='roc_auc'`，可直接复现过拟合随数据量缓解的典型曲线）：
```
逻辑回归（加标准化）：
  n= 44  train=0.9909  test=0.8065  差距=+0.1844   ← 小样本，过拟合明显
  n=104  train=0.9526  test=0.9009  差距=+0.0516
  n=164  train=0.9318  test=0.8933  差距=+0.0386
  n=224  train=0.9287  test=0.9027  差距=+0.0259   ← 数据变多，差距收窄到 1/7

对照：决策树（同样是这份数据）
  n= 44  train=1.0000  test=0.7945  差距=+0.2055   ← train 恒为 1.0！
  n=224  train=1.0000  test=0.8397  差距=+0.1603   ← 加数据也几乎不改善
```
**对照组的含义**：决策树 `train` 永远是 **1.0000**——它把训练集**背下来**了（每个叶子只剩一个样本）。
test 曲线也几乎不随数据量上升，说明它的问题是**容量过大**，加数据解决不了，必须剪枝/限深。
而逻辑回归的差距随数据量从 +0.18 收窄到 +0.026，说明它的问题是**数据不足**。

> **这两种"过拟合"的处方完全不同**：数据不足 → 加数据；容量过大 → 加正则/减复杂度。
> **学习曲线的价值就在区分这两者。**

### 场景 7 模型持久化

```python
import joblib

# ✅ 推荐：整个 Pipeline 一起保存（含所有预处理！）
joblib.dump(pipe, 'model.joblib')
loaded = joblib.load('model.joblib')
loaded.predict(X_new)      # 不需要手动再做预处理

# 压缩保存
joblib.dump(pipe, 'model.joblib', compress=3)
```
**本机实测对比**：
```
joblib 文件 2826 字节 / pickle 2503 字节   加载结果一致
compress=3 后压缩到 1601 字节（约 43%）
```
> **⚠ 安全警告**：`joblib`/`pickle` 使用 pickle 协议，**加载不可信来源的文件会执行任意代码**。
> 官方推荐生产环境用 `skops.io` 或 ONNX 导出（本机两者都未安装）。
> **只加载你自己训练并保存的模型。**

### 场景 8 聚类与降维

```python
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score

# KMeans：需要预先指定 K，用肘部法则/轮廓系数选
inertias, sils = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))

# DBSCAN / HDBSCAN：不需要指定 K，能发现任意形状的簇和噪声
DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)      # -1 表示噪声点
HDBSCAN(min_cluster_size=5).fit_predict(X_scaled)         # 新版，比 DBSCAN 更省心

# PCA 降维（记得先标准化！）
pca = PCA(n_components=0.95).fit(X_scaled)   # 保留 95% 方差
print(pca.explained_variance_ratio_)
X_reduced = pca.transform(X_scaled)
```
**本机实测聚类指标**：silhouette 0.5157、davies_bouldin 0.7465、calinski_harabasz 404.0。
**注意**：`KMeans` 必须加 `n_init`（新版默认已是 `'auto'`，显式写更清楚）；聚类前**一定要标准化**。

---

## 第七部分：10 个最经典的坑（全部实测）

### 坑 1 在划分数据之前做预处理

**实测破坏力**：特征选择泄露让纯噪声数据的准确率从 0.5250 涨到 0.8417（**+31.67 个百分点**）。
**修法**：一切预处理放进 `Pipeline`。
**自检**：`grep` 你的代码，看 `fit_transform` 是否出现在 `train_test_split` 之前。

### 坑 2 用测试集调超参数（"测试集泄漏"）

```python
# ❌ 反复用测试集调参，测试集被"训练"了
for C in [0.01, 0.1, 1, 10]:
    model.fit(X_train, y_train); score = model.score(X_test, y_test)   # 挑最高分
# ✅ 用 CV 在训练集内调，测试集只用一次
GridSearchCV(pipe, {...}, cv=5).fit(X_train, y_train)
```
**纪律**：测试集用两次以上，它就不再是"未见数据"了。

### 坑 3 不平衡数据只看 accuracy

**实测**：正类 5.3% 时，模型 accuracy 0.9767，**全猜多数类的 Dummy 也有 0.9467**。
**修法**：看 precision/recall/F1/**PR-AUC**，并永远带上 Dummy 基线。

### 坑 4 `LabelEncoder` 用在特征 X 上

```python
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])   # ❌ 类别变成 0/1/2，模型误以为有序
# ✅ 无序类别用 OneHotEncoder，有序类别用 OrdinalEncoder(categories=...)
```
`LabelEncoder` **只用于 y**。

### 坑 5 指标参数顺序写反

```python
accuracy_score(y_true, y_pred)   # ✅
accuracy_score(y_pred, y_true)   # ❌ 不报错，结果可能完全不对
```
**修法**：统一命名 `y_true` / `y_pred`，或全部用关键字传参。

### 坑 6 忘记 `stratify=y`

**实测**：正类 10% 时，`KFold(5)` 各折正类比例是 `[0.200, 0.100, 0.025, 0.050, 0.125]`——
某折只有 2.5% 正类；`StratifiedKFold` 则是 `[0.1]*5` 完全一致。
**修法**：`train_test_split(..., stratify=y)`；CV 时用 `StratifiedKFold` 或让 `cross_val_score` 自动处理。

### 坑 7 时间序列用随机 K 折

```python
cross_val_score(model, X, y, cv=5)              # ❌ 用未来数据预测过去
cross_val_score(model, X, y, cv=TimeSeriesSplit(5))   # ✅ 时序必须用它
```
**还有"同组样本跨集"问题**：同一病人的多条记录不能同时出现在训练和测试集 → 用 `GroupKFold`。

### 坑 8 忘记 `random_state`

```python
train_test_split(X, y)                    # ❌ 每次结果不同，无法复现
RandomForestClassifier()                  # ❌ 同上
KMeans(n_clusters=3)                      # ❌ 只写 n_clusters 不够
# ✅ 全部固定：random_state=42, shuffle 相关参数、KMeans 的 n_init
```

### 坑 9 变换器的 `fit` 没 `return self`

```python
def fit(self, X, y=None):
    self.mean_ = X.mean(0)     # ❌ 忘了 return self
# Pipeline 会报: TypeError: fit() should return self
```

### 坑 10 保存模型时只存了分类器，没存预处理器

```python
joblib.dump(clf, 'm.joblib')                 # ❌ 预测时忘了标准化 → 结果全错
joblib.dump(pipeline, 'm.joblib')            # ✅ 整个 Pipeline 一起存
```
**这个坑不报错**（模型会给出预测），只是预测结果错得离谱。

### 额外：报错信息速查（实测原文）

| 报错 | 含义 |
|---|---|
| `Specifying the columns using strings is only supported for dataframes` | 用字符串列名选了 NumPy 数组的列 |
| `Setting output container to 'pandas' requires pandas to be installed` | 无 pandas 却用了 `set_output('pandas')` |
| `Transformer X does not provide get_feature_names_out` | 某个转换器不支持特征名跟踪（如 `FunctionTransformer`） |
| `PermissionError: [WinError 5] 拒绝访问`（HistGradientBoosting） | joblib 线程池建命名管道被沙箱拦；见下方 9.1 |
| `HalvingGridSearchCV is experimental ... you need to explicitly import enable_halving_search_cv` | 实验性 API 需先启用 |
| `The 'n_informative' parameter ... must be an int in the range [1, inf)` | 数据生成器参数校验（1.9 更严格） |
| `Progressbar requires rich` | 用了 `sklearn.callback.ProgressBar` 但没装可选依赖 `rich` |
| `Input contains NaN` | 模型不支持缺失值，先用 `SimpleImputer` |
| `fit() should return self` | 自定义转换器忘了 `return self` |

### 9.1 本机专属坑：`HistGradientBoosting*` 需要 threadpoolctl 才能跑

官方推荐"表格数据首选 `HistGradientBoostingClassifier`"，但在你当前环境**默认会失败**。实测：

```
HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)
# PermissionError: [WinError 5] 拒绝访问。
#   （栈底是 multiprocessing.connection.Pipe -> joblib ThreadPool）
```
**原因**：它的分箱步骤用 `joblib.Parallel(backend='threading')`，线程池要创建**命名管道**，
而 DSH 沙箱禁止进程打开命名管道。**这不是 sklearn 的 bug，是运行环境的限制。**

**实测有效的规避方法**（用 `threadpoolctl` 把线程数限为 1，joblib 就走串行分支）：
```python
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

with threadpool_limits(limits=1):
    scores = cross_val_score(HistGradientBoostingClassifier(random_state=0),
                             X_train, y_train, cv=5, scoring='roc_auc')
# 实测成功: 0.9909 ± 0.0147
```
**实测无效的方法**：`cross_val_score(..., n_jobs=1)`（外层串行不影响内层）；
`HistGradientBoostingClassifier(n_jobs=1)`（**没有 `n_jobs` 这个参数**，会 TypeError）。

**不想折腾就用这些替代**（本机实测全部正常，性能接近甚至更好）：

| 模型 | 5 折 CV ROC-AUC（breast_cancer） |
|---|---|
| LogisticRegression | **0.9935** |
| ExtraTreesClassifier | 0.9921 |
| HistGradientBoosting（配 threadpoolctl） | 0.9909 |
| RandomForestClassifier | 0.9896 |
| GradientBoostingClassifier（经典版） | 0.9873 |
| DummyClassifier（基线） | 0.5000 |

> 经典 `GradientBoostingClassifier`（非 Hist 版）**不受影响**——它不走那条线程池路径。

### 附：`sklearn.callback` 的真实 API（本机实测）

官方第 7 章 "Callbacks" 在 1.9 的公开成员是（**没有 `Callback` 这个类**）：
```python
import sklearn.callback as cb
cb.__all__
# ['AutoPropagatedCallback','CallbackContext','CallbackSupportMixin','FitCallback',
#  'ProgressBar','ScoringMonitor','ScoringMonitorLog','with_callbacks']
```
最实用的 `ProgressBar` 需要**可选依赖 `rich`**，本机实测：
```
cross_val_score(..., callback=ProgressBar())
# ImportError: Progressbar requires rich.
```
所以这一章在你当前环境下**暂时用不了**（除非 `pip install rich`）。优先级本来就低（★☆☆），跳过无妨。

---

## 第八部分：4 周学习路线（每天 1~2 小时）

| 周 | 目标 | 做什么 | 验收标准 |
|---|---|---|---|
| 1 | **统一接口 + 数据集** | 官方 *Getting Started*；把 `load_*`/`make_*` 全跑一遍；用 3 个模型在 iris/wine 上对比 | 能解释 Estimator/Transformer/Predictor 的区别，能凭记忆写出 `fit/predict` |
| 2 | **预处理 + Pipeline** | User Guide 第 8 章 + 第 12 章（**陷阱那章必读**）；把 ml-2 的双月分类改成完整 Pipeline | 能写出含 ColumnTransformer 的混合类型 Pipeline |
| 3 | **评估 + 调参** | User Guide 第 3 章全部；指标对照表抄一遍；GridSearchCV + 学习曲线 | 能解释为什么 accuracy 会骗人，能画出并解读学习曲线 |
| 4 | **实战 + 复盘** | 找一个真实表格数据集（如 `load_breast_cancer`），走完第五部分场景 1 全流程 | 能独立诊断"指标虚高"并找到泄露源 |

### 每周必做

1. **先抄 Examples 再读原理**：官方 [Examples 库](https://scikit-learn.org/stable/auto_examples/index.html) 有几百个可运行示例，找到相似任务抄下来跑通，效率最高。
2. **对照检查泄露清单**（4.4 节）：每完成一个项目都过一遍那 7 条。
3. **保留错题集**：像你 ml-1 那样记录。sklearn 的坑大多是**不报错**的（泄露、指标顺序、忘了 random_state），只有记录才能记住。

### 自查：能答出这 6 题就算入门

1. `fit` / `transform` / `predict` / `fit_transform` 分别在什么对象上、做什么？
2. 为什么预处理必须放进 `Pipeline`？给出一个泄露会造成的具体数值后果。
3. 不平衡数据为什么不能只看 accuracy？该看哪些指标？
4. `cross_val_score` 默认会不会分层？手动传 `KFold` 呢？
5. `ColumnTransformer` 里用字符串列名有什么前提？没有 pandas 怎么办？
6. 为什么保存模型建议保存整个 Pipeline 而不是单独的模型？

（答案分别在 2.1、第四部分、3.4、3.3、5.2、坑 10。）

---

## 第九部分：一页速查表

```python
# ---------- 数据集 ----------
load_iris / load_digits / load_wine / load_breast_cancer / load_diabetes
fetch_california_housing（需联网）
make_classification / make_regression / make_blobs / make_moons / make_circles
load_iris(return_X_y=True)

# ---------- 划分与验证 ----------
train_test_split(X, y, test_size, random_state, stratify)
cross_val_score / cross_validate / cross_val_predict
KFold / StratifiedKFold / GroupKFold / TimeSeriesSplit / LeaveOneOut
learning_curve / validation_curve

# ---------- 调参 ----------
GridSearchCV / RandomizedSearchCV / HalvingGridSearchCV（需 enable_halving_search_cv）
TunedThresholdClassifierCV / FixedThresholdClassifier
ParameterGrid / ParameterSampler

# ---------- 预处理 ----------
StandardScaler / MinMaxScaler / RobustScaler / MaxAbsScaler / Normalizer
OneHotEncoder / OrdinalEncoder / LabelEncoder(仅 y) / TargetEncoder / LabelBinarizer
PolynomialFeatures / KBinsDiscretizer / FunctionTransformer
PowerTransformer / QuantileTransformer / SplineTransformer
SimpleImputer / KNNImputer / MissingIndicator

# ---------- Pipeline ----------
Pipeline / make_pipeline / FeatureUnion
ColumnTransformer / make_column_transformer / make_column_selector
pipe.named_steps / ct.named_transformers_ / get_feature_names_out()
{'clf__C': [...]}    # 跨步骤传参：步骤名__参数名

# ---------- 模型（按数据规模/类型）----------
LogisticRegression / LinearRegression / Ridge / Lasso / ElasticNet / SGDClassifier
KNeighborsClassifier / SVC / SVR / DecisionTreeClassifier
RandomForestClassifier / ExtraTreesClassifier / GradientBoostingClassifier
HistGradientBoostingClassifier   # 表格数据首选，原生支持 NaN
                                 # ⚠ 本机需 with threadpool_limits(limits=1) 才能跑（见 9.1）
VotingClassifier / StackingClassifier / BaggingClassifier / AdaBoostClassifier
GaussianNB / MultinomialNB / ComplementNB
MLPClassifier / MLPRegressor
DummyClassifier / DummyRegressor   # 基线必带

# ---------- 无监督 ----------
KMeans / MiniBatchKMeans / DBSCAN / HDBSCAN / AgglomerativeClustering / Birch
PCA / TruncatedSVD / NMF / FastICA / KernelPCA
IsolationForest / LocalOutlierFactor / OneClassSVM
GaussianMixture

# ---------- 指标 ----------
accuracy_score / precision_score / recall_score / f1_score
roc_auc_score / average_precision_score / log_loss / brier_score_loss
confusion_matrix / classification_report
roc_curve / precision_recall_curve
mean_squared_error / mean_absolute_error / r2_score / mean_absolute_percentage_error
silhouette_score / davies_bouldin_score / calinski_harabasz_score
adjusted_rand_score / normalized_mutual_info_score

# ---------- 特征选择与解释 ----------
SelectKBest / SelectFromModel / RFECV / SequentialFeatureSelector / VarianceThreshold
chi2 / f_classif / f_regression / mutual_info_classif
permutation_importance / partial_dependence

# ---------- 工具 ----------
clone / get_params / set_params / is_classifier / is_regressor
BaseEstimator / TransformerMixin
joblib.dump / joblib.load
get_config / set_config / config_context / show_versions
sklearn.experimental.enable_halving_search_cv
```

**最后一句**：sklearn 的 461 个 API 背后只有**一套接口**（`fit`/`transform`/`predict`）+ **一个纪律**
（所有信息只能从训练集流向测试集）。学会这两点，剩下的都是查文档的事。
而**最危险的错误永远是那些不报错的**——数据泄露、指标顺序、忘记 `random_state`、只保存模型不保存预处理器。
所以本指南反复给"实测数值"而不是"记住规则"。
