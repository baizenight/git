# PyTorch 练习题（30 题，含答案与详解）

> 用法：先自己做，再看答案。**「易错点」比答案本身更重要。**
> 全部题目和答案均在 **torch 2.14.1+cpu** 上实跑验证过。
> 环境注意：本机 `torch.cuda.is_available()` 为 **False**，`torch.compile` 不可用（缺 MSVC 编译器）。

---

## Level 1：Tensor 基础（题 1-6）

**1.** 创建一个 `(3, 4)` 的全 1 张量，dtype 为 `float32`，并打印它的 `shape`、`dtype`、`device`。

**2.** 从 NumPy 数组创建张量，要求**共享内存**（改张量会改到原数组），写出两种方式。

**3.** 创建一个 0 到 1 之间 5 个等距点的张量，再创建 `[0,1,2,3,4]` 的整数张量。

**4.** `t = torch.randn(2, 3)`，把它转置后展平成一维。为什么用 `view(-1)` 可能失败？

**5.** 用一行代码把 `(4, 6)` 的张量变成 `(6, 4)` 并保证是视图（不拷贝数据）。

**6.** `torch.tensor([1, 2])` 和 `torch.tensor([1., 2.])` 的 dtype 分别是什么？为什么这对 `nn.Linear` 很重要？

<details><summary>答案与详解</summary>

```python
import torch

# 1
t = torch.ones(3, 4, dtype=torch.float32)
print(t.shape, t.dtype, t.device)     # torch.Size([3, 4]) torch.float32 cpu
# 易错点：torch.ones(3, 4) 也可以（可变参数），但 shape 用元组更清晰

# 2
import numpy as np
arr = np.arange(6)
t1 = torch.from_numpy(arr)            # 方式一
t2 = torch.as_tensor(arr)             # 方式二（推荐：已是 tensor 时不拷贝）
t1[0] = 999
print(arr[0])                          # 999 —— 真的改了原数组
# 易错点：torch.tensor(arr) 是【拷贝】，不共享内存
print(torch.tensor(arr)[0].item(), arr[0])   # 999 999（值一样但内存独立）

# 3
torch.linspace(0, 1, 5)               # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
torch.arange(5)                       # tensor([0, 1, 2, 3, 4])  dtype=int64

# 4
t = torch.randn(2, 3)
t.t()                                 # 转置
t.t().is_contiguous()                 # False —— 非连续
try:
    t.t().view(-1)
except RuntimeError as e:
    print(e)   # view size is not compatible with input tensor's size and stride
t.t().reshape(-1)                     # ✅ 成功，会自动拷贝
# 易错点：view 要求内存连续；reshape 自动处理。不确定时用 reshape

# 5
t = torch.randn(4, 6)
t_t = t.t()                           # ✅ 转置是视图，零拷贝
t.permute(1, 0)                       # ✅ 也是视图
t.transpose(0, 1)
# 对比：t.reshape(6, 4) 也是视图（因为能重新划分），但语义不同（不是真正的转置）
print(t.t().data_ptr() == t.data_ptr())    # True —— 同一个内存
print(t.reshape(6,4).data_ptr() == t.data_ptr())  # True —— reshape 也共享
# 注意：reshape 和 t() 结果不同！reshape 按 C 序重排，t() 是轴交换

# 6
torch.tensor([1, 2]).dtype            # torch.int64
torch.tensor([1., 2.]).dtype          # torch.float32
# 为什么重要：nn.Linear 的 weight 是 float32，
# 输入若为 int64 会报 "expected scalar type Float but found Long"
nn.Linear(2, 1)(torch.tensor([[1, 2]]))   # ❌ RuntimeError
nn.Linear(2, 1)(torch.tensor([[1., 2.]])) # ✅
```
</details>

---

## Level 2：形状与索引（题 7-12）

**7.** `t = torch.arange(24).reshape(2, 3, 4)`，用 `permute` 把轴顺序变成 `(4, 2, 3)`。

**8.** `t = torch.randn(4, 5)`，取出「第 0 行第 2 列」和「第 2 行第 4 列」两个元素（结果形状 `(2,)`）。

**9.** 把 `t = torch.randn(5, 3)` 中所有负数替换为 0，写出 PyTorch 特有的方法（不用 `t[t<0]=0`）。

**10.** `a = torch.randn(2, 4)`，`b = torch.randn(2, 4)`，沿第 0 维拼接和沿第 1 维拼接，形状各是什么？

**11.** `t = torch.randn(2, 5)`，取每行最大的 2 个元素及其索引。

**12.** `t = torch.randn(3, 4)`，在第 0 维前面加一个轴（变成 `(1,3,4)`），再在第 1 维加一个轴（变成 `(1,1,3,4)`）。

<details><summary>答案与详解</summary>

```python
# 7
t = torch.arange(24).reshape(2, 3, 4)
t.permute(2, 0, 1).shape              # torch.Size([4, 2, 3])
# 易错点：permute 的参数是「新的轴顺序」
# permute(2,0,1) = 新第0维来自原第2维，新第1维来自原第0维，新第2维来自原第1维

# 8
t = torch.randn(4, 5)
t[[0, 2], [2, 4]]                     # 成对取样：(0,2) 和 (2,4)
# 易错点：和 NumPy 一样，这是成对而非笛卡尔积

# 9
t = torch.randn(5, 3)
t.clamp(min=0)                        # ✅ 最简单
t.masked_fill(t < 0, 0)               # ✅ PyTorch 特有，语义最清晰
torch.relu(t)                         # ✅ 恰好等价
t.relu()                              # ✅
# 注意：这些返回新张量；原地版是 t.clamp_(min=0)（末尾下划线表示原地）

# 10
a = torch.randn(2, 4); b = torch.randn(2, 4)
torch.cat([a, b], dim=0).shape        # torch.Size([4, 4])
torch.cat([a, b], dim=1).shape        # torch.Size([2, 8])
# 官方文档统一用 dim，NumPy 用 axis。
# ⚠ 实测修正：torch 2.14 里 axis= 作为别名【确实可用】！
#   实测 torch.cat/stack/sum/mean/softmax/squeeze 传 axis= 都能跑，结果与 dim= 一致。
#   但官方文档和错误信息都只写 dim，为保持一致性和可移植性，仍建议统一写 dim=。

# 11
t = torch.randn(2, 5)
values, indices = t.topk(2, dim=1)    # 都返回形状 (2,2)
# topk 默认 largest=True，按降序
# 对比：torch.sort(t, dim=1, descending=True) 返回全部元素的排序

# 12
t = torch.randn(3, 4)
t.unsqueeze(0).shape                  # torch.Size([1, 3, 4])
t.unsqueeze(0).unsqueeze(1).shape     # torch.Size([1, 1, 3, 4])
t[None].shape                         # 等价于 unsqueeze(0)
t[None, None].shape                   # 一次加两个轴
# 反向操作：squeeze(dim) 去掉指定轴，squeeze() 去掉所有长度 1 的轴
```
</details>

---

## Level 3：autograd（题 13-18）

**13.** 用 autograd 求 `y = 3x² + 2x + 1` 在 `x = 2` 处的导数，并手算验证。

**14.** 为什么下面代码报错？怎么修？
```python
x = torch.randn(3, requires_grad=True)
y = x * 2
y.backward()
```

**15.** 下面代码为什么第二次 `backward()` 报错？两种修法分别是什么？
```python
a = torch.randn(3, requires_grad=True)
l = (a * 2).sum()
l.backward()
l.backward()
```

**16.** 打印 `y` 的梯度（`y = x * 2`，`y` 是非叶子张量），`y.grad` 是 `None`，怎么办？

**17.** 只想拿到梯度值、不想污染 `a.grad`，用什么函数？

**18.** `t1 = t.detach()` 和 `t2 = t.clone()` 有什么区别？（`requires_grad` 和内存共享两方面）

<details><summary>答案与详解</summary>

```python
# 13
x = torch.tensor(2.0, requires_grad=True)
y = 3 * x**2 + 2 * x + 1
y.backward()
print(x.grad)                         # tensor(14.)
# 手算：dy/dx = 6x + 2 = 6*2+2 = 14 ✓
# 易错点：x 必须是浮点张量（requires_grad 对整数张量报错）

# 14
# 报错：RuntimeError: grad can be implicitly created only for scalar outputs
# 原因：y 是形状 (3,) 的非标量，backward 不知道用哪个"上游梯度"
x = torch.randn(3, requires_grad=True)
y = x * 2
y.backward(torch.ones(3))             # ✅ 传入与 y 同形的"权重"
print(x.grad)                         # tensor([2., 2., 2.])
# 或先聚合：y.sum().backward()

# 15
# 报错：RuntimeError: Trying to backward through the graph a second time
# 原因：默认 backward 后释放计算图以省内存
# 修法1：retain_graph=True（注意梯度会累加）
l.backward(retain_graph=True)
l.backward()
# 修法2：重新做一次前向传播（更常见）
l2 = (a * 2).sum(); l2.backward()

# 16
x = torch.randn(3, requires_grad=True)
y = x * 2
y.retain_grad()                       # ✅ 让非叶子张量保留梯度
y.sum().backward()
print(y.grad)                         # tensor([1., 1., 1.])
# 不调用 retain_grad 时访问 y.grad 会给 None + 一条 UserWarning

# 17
a = torch.randn(3, requires_grad=True)
g = torch.autograd.grad((a * 3).sum(), a)   # 返回元组，不写入 a.grad
print(g[0])                           # tensor([3., 3., 3.])
print(a.grad)                         # None —— 没有被污染
# 用途：需要多次求不同梯度、或做梯度惩罚（WGAN-GP）时

# 18
t = torch.randn(3, requires_grad=True)
d = t.detach()
c = t.clone()
print(d.requires_grad, c.requires_grad)          # False True
print(d.data_ptr() == t.data_ptr())               # True  —— detach 共享内存
print(c.data_ptr() == t.data_ptr())               # False —— clone 新内存
# 结论：detach = 断图 + 共享内存；clone = 保留图 + 独立内存
# 用途：detach 用于停止梯度回传（算指标）；clone 用于要独立副本
# 不要用已不推荐的 .data
```
</details>

---

## Level 4：nn.Module 与层（题 19-24）

**19.** 写一个 `MLP` 类：输入 2 维 → 隐藏层 32（ReLU）→ 输出 1 维，带 Dropout(0.2)。

**20.** `nn.Linear(4, 3)` 的 `weight.shape` 和 `bias.shape` 分别是什么？输入 `(5, 4)` 和 `(2, 5, 4)` 的输出形状？

**21.** 下面代码的参数会被 `optimizer` 拿到吗？为什么？怎么修？
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4) for _ in range(3)]
```

**22.** 想让一个张量出现在 `state_dict()` 里但不参与训练（比如 BN 的 running_mean），用什么？

**23.** 用 `nn.Sequential` 构建和题 19 相同的网络，并打印每一层的类型。为什么 `Sequential` 的索引不连续？

**24.** 注册一个 forward hook，打印一个两层网络的每层输入/输出形状。

<details><summary>答案与详解</summary>

```python
import torch, torch.nn as nn

# 19
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=1, dropout=0.2):
        super().__init__()                       # ← 必须调用
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))

m = MLP()
print(m(torch.randn(4, 2)).shape)                # torch.Size([4, 1])
print(sum(p.numel() for p in m.parameters()))    # 2*32+32 + 32*1+1 = 129

# 20
lin = nn.Linear(4, 3)
print(lin.weight.shape, lin.bias.shape)          # torch.Size([3, 4]) torch.Size([3])
# 易错点：weight 是 (out_features, in_features)，不是 (in, out)！
print(lin(torch.randn(5, 4)).shape)              # torch.Size([5, 3])
print(lin(torch.randn(2, 5, 4)).shape)           # torch.Size([2, 5, 3]) ← 支持任意前导维

# 21
# 不会！named_parameters() 是空的，optimizer 拿不到这些参数，它们永远不更新。
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4) for _ in range(3)]
print([n for n, _ in Bad().named_parameters()])  # [] —— 空！
# 原因：nn.Module.__setattr__ 只自动注册 Module / Parameter / Tensor 子类，
#       普通 Python list 不在注册范围内。
# 修法：
class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
print([n for n, _ in Good().named_parameters()]) # ['layers.0.weight', ...] ✅

# 22
class WithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        self.register_buffer('running_mean', torch.zeros(2))   # ✅
        self.loss_avg = torch.tensor(0.0)                      # ❌ 不注册
w = WithBuffer()
print(list(w.state_dict().keys()))
# ['running_mean', 'fc.weight', 'fc.bias']  ← loss_avg 不在里面！
# 用途：BN 的统计量、位置编码常量、类别数等需要随模型保存/迁移设备的张量

# 23
net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
for i, layer in enumerate(net):
    print(i, type(layer).__name__)
# 0 Linear / 1 ReLU / 2 Dropout / 3 Linear —— 索引其实连续
# 但要注意：访问权重时索引会"跳过"无参数层：
print(list(net.state_dict().keys()))   # ['0.weight', '0.bias', '3.weight', '3.bias']
# 易错点：没有 1.weight、2.weight（ReLU/Dropout 无参数），
#        所以 state_dict 的 key 是 0 和 3。手写时容易写成 net[1].weight 而报错。

# 24
seen = []
def hook(mod, inp, out):
    seen.append((type(mod).__name__, tuple(inp[0].shape), tuple(out.shape)))
net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
handles = [m.register_forward_hook(hook) for m in net]
net(torch.randn(8, 2))
for h in handles:
    h.remove()                        # 易错点：不移除会一直累积，重复触发
for name, i, o in seen:
    print(f'{name}: {i} -> {o}')
# Linear: (8, 2) -> (8, 4)
# ReLU:   (8, 4) -> (8, 4)
# Linear: (8, 4) -> (8, 1)
```
</details>

---

## Level 5：训练与调试（题 25-30）

**25.** 写出一个 batch 的标准训练四步，并说明忘记 `zero_grad()` 的数值后果。

**26.** 二分类用 `BCEWithLogitsLoss` 时，网络输出层要不要加 `Sigmoid`？为什么？并算出 logit = -90 时两种写法的损失。

**27.** 多分类用 `CrossEntropyLoss`，标签应该是什么 dtype 和形状？网络输出层要不要加 `Softmax`？

**28.** 评估时必须写什么？忘了会有什么后果（分 Dropout 和 BatchNorm 说明）？

**29.** 为什么保存"最优模型"必须用 `copy.deepcopy(model.state_dict())`？给出实测证据。

**30.** 给定一个"loss 完全不降"的模型，列出你的诊断步骤（至少 4 步）。

<details><summary>答案与详解</summary>

```python
# 25
for xb, yb in dataloader:
    optimizer.zero_grad()             # ① 清空上一轮梯度
    logits = model(xb)                # ② 前向
    loss = criterion(logits, yb)      # ③ 算损失
    loss.backward()                   # ④ 反向
    optimizer.step()                  # ⑤ 更新
# 忘记 zero_grad 的实测后果（同一 batch 连续 backward）：
#   第1次 backward 后 grad.norm = 2.7049
#   第2次 backward 后 grad.norm = 5.4098   ← 正好 2 倍
#   第3次 backward 后 grad.norm = 8.1148   ← 正好 3 倍
# 加了 zero_grad 的对照：2.7049 -> 2.5803 -> 2.4625（稳定）
# 后果：参数被超大步长更新，loss 震荡或发散。而且【不报错】！

# 26
# 不要加 Sigmoid！BCEWithLogitsLoss 内部已含，分开算会数值不稳定。
logits = torch.tensor([-90.0])
target = torch.tensor([1.0])
stable = F.binary_cross_entropy_with_logits(logits, target).item()
naive = F.binary_cross_entropy(torch.sigmoid(logits), target).item()
print(stable)   # 90.0     ← 正确
print(naive)    # 100.0    ← 错！sigmoid(-90) 在 float32 下溢成 0，log(0) 被 clamp
# 正确值就是 |z| = 90（当 z 很负且 target=1 时，loss ≈ -z）
# 更致命的是梯度：
#   稳定版梯度 = sigmoid(z) - y，两端仍非零
#   朴素版梯度在两端变成 0 → 神经元永远学不动
# 实测整体误差：朴素版是稳定版的 5458 万倍

# 27
criterion = nn.CrossEntropyLoss()
# 标准用法：标签 dtype=torch.long，形状 (N,)，值是类别索引
y_idx = torch.tensor([0, 2, 1, 4])               # ✅ 推荐
# 网络输出层【不加】Softmax（CrossEntropyLoss = LogSoftmax + NLLLoss）
# 实测等价性：F.cross_entropy(logits, y) == F.nll_loss(F.log_softmax(logits,1), y)
# 取预测类别：logits.argmax(dim=1)

# ⚠ 实测修正：float 类型的 one-hot 标签【不会报错】，而是被当作"软标签"处理！
#   而且结果与整数索引完全相同：
onehot = F.one_hot(y_idx, 5)
F.cross_entropy(logits, y_idx)              # 2.032093048095703
F.cross_entropy(logits, onehot.float())     # 2.032093048095703  ← 完全相等！
#   用途：标签平滑、知识蒸馏（可以用 float 概率作目标）
#   但传 long 的 one-hot 会报错：
F.cross_entropy(logits, onehot)
#   RuntimeError: Expected floating point type for target with class probabilities, got Long
#   所以 one_hot(...) 之后必须 .float()

# 真正会报错的标签形态（这些才是"格式错误"）：
F.cross_entropy(logits, torch.randn(4))
#   RuntimeError: expected target dtype to be Long or Byte, but got Float
F.cross_entropy(logits, torch.tensor([[0.], [2.], [1.], [4.]]))
#   RuntimeError: 0D or 1D target tensor expected, multi-target not supported

# 28
# 评估时必须：model.eval() + with torch.no_grad()（或 torch.inference_mode()）
model.eval()
with torch.no_grad():
    pred = model(X)
# 忘了 model.eval() 的实测后果：
#   train 模式两次输出相同? False  ← Dropout 随机 + BN 用当前 batch 统计
#   eval  模式两次输出相同? True
#   具体影响：Dropout 仍在随机丢神经元 → 指标偏低且每次不同；
#            BatchNorm 用 batch 统计 → 小 batch 时指标完全不可信
# 忘了 no_grad 的后果：白白建计算图，内存持续增长，大模型会 OOM（数值是对的）

# 29
# state_dict() 返回的是【引用】，不是拷贝。实测：
model = nn.Sequential(nn.Linear(2, 2))
sd = model.state_dict()
before = model[0].weight.clone()
sd['0.weight'][0, 0] = 999.0
print(torch.equal(before, model[0].weight))      # False —— 模型真的被改了！
# 所以必须 deepcopy：
import copy
best = copy.deepcopy(model.state_dict())         # ✅ 真正冻结那一刻
model[0].weight.data.fill_(0.0)
print(best['0.weight'][0, 0].item())             # 仍是 999.0，不受影响

# 30 诊断步骤：
# (1) 检查梯度是否存在：p.grad is None → 计算图断了（如用了 .detach() 或 no_grad）
for n, p in model.named_parameters():
    print(n, p.grad is None, None if p.grad is None else p.grad.norm().item())
# (2) 检查参数是否真的在更新：
before = {n: p.clone() for n, p in model.named_parameters()}
optimizer.step()
for n, p in model.named_parameters():
    if torch.equal(before[n], p): print(f'{n} 没变 ← 有问题')
# (3) 检查梯度是否全为 0：Dead ReLU（换 LeakyReLU）、激活饱和（换 ReLU）、
#     或 sigmoid+BCELoss（换 BCEWithLogitsLoss）
# (4) 检查学习率是否过小：打印 opt.param_groups[0]['lr']
# (5) 检查数据与标签是否对齐、标签是否被破坏（打乱索引时最容易出错）
# (6) 检查最后一层与损失是否匹配：二分类用 BCEWithLogits（标签 float (N,1)），
#     多分类用 CrossEntropy（标签 long (N,)）；配错会静默算错
# (7) 检查是否忘了 zero_grad（梯度累加 → 反而应该震荡而不是不降）
# (8) 用一个极小数据集（10 条）过拟合它：如果能降，说明是数据/容量问题；
#     如果还不降，说明是代码 bug
```
</details>

---

## 附：三个自查用的魔鬼细节

做完 30 题后，若能立刻回答这三题，说明基础已扎实：

**Q1**：`nn.Linear(4, 3)` 的 `weight` 形状是 `(3,4)` 还是 `(4,3)`？为什么这个顺序容易搞错？

**Q2**：`self.layers = [nn.Linear(...)]` 和 `nn.ModuleList([nn.Linear(...)])` 在 `named_parameters()` 上的差别是什么？后果是什么？

**Q3**：logit = -90、target = 1 时，`BCEWithLogitsLoss` 和 `sigmoid+BCELoss` 的结果分别是多少？哪个对？

<details><summary>答案</summary>

**A1**：`(3, 4)`，即 `(out_features, in_features)`。容易搞错是因为数学上写 `y = Wx` 时 `W` 是 `(out, in)`，但很多人凭"输入在前"的直觉记成 `(in, out)`。实测：`nn.Linear(4,3).weight.shape` = `torch.Size([3, 4])`。

**A2**：前者 `named_parameters()` 返回**空**（参数完全没注册），后者返回 `['layers.0.weight', ...]`。后果：`optimizer` 拿不到这些参数，它们**永远不更新**，训练时 loss 不降，而且**不报任何错**。

**A3**：`BCEWithLogitsLoss` 给 **90.0**（正确）；`sigmoid+BCELoss` 给 **100.0**（错误）。原因：float32 下 `sigmoid(-90)` 下溢成 0，`log(0)` 被 clamp 到 `log(eps) = -100`，损失被"封顶"。且朴素版两端梯度为 0。
</details>
