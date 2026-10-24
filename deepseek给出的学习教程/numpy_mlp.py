"""纯 NumPy 实现的两层 MLP —— 不用 autograd，手写前向/反向传播。

用途：验证「我是否真的懂 MLP 的构建」。
看完这个文件后，请自己不看代码重写一遍，跑出同样的 100% 准确率。

结构：2 -> H -> 1，隐藏层 ReLU，输出层二分类（BCEWithLogits）
PyTorch 里的等价写法见文件末尾注释。
"""

import numpy as np

# ----------------------------------------------------------------------
# 1. 数据：XOR（单层感知机解决不了的经典问题）
#    再加两个同分布样本，便于可视化决策边界
# ----------------------------------------------------------------------
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])

# ----------------------------------------------------------------------
# 2. 初始化：He 初始化（对应 nn.init.kaiming_normal_）
#    为什么方差是 2/n_in？ReLU 把负半轴砍掉，方差减半，所以要补回来一倍。
#    试试把它改成 * 0.01 或 * 10，观察训练是否变差——这就是「初始化」的直观理解。
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)
H = 8
W1 = rng.standard_normal((H, 2)) * np.sqrt(2.0 / 2)   # (H, in)
b1 = np.zeros((1, H))
W2 = rng.standard_normal((1, H)) * np.sqrt(2.0 / H)   # (out, H)
b2 = np.zeros((1, 1))

lr = 0.5
for epoch in range(2001):
    n = X.shape[0]

    # ---------------- 前向传播 ----------------
    Z1 = X @ W1.T + b1          # (n, H)  隐藏层线性部分
    A1 = np.maximum(Z1, 0)      # (n, H)  ReLU，引入非线性全靠它
    Z2 = A1 @ W2.T + b2         # (n, 1)  输出层 logits（不接 Sigmoid）

    # BCEWithLogits：数值稳定的写法，避免 log(0)
    # loss = -mean( y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z)) )
    #      = mean( max(z,0) - z*y + log(1+exp(-|z|)) )
    loss = np.mean(np.maximum(Z2, 0) - Z2 * y + np.log1p(np.exp(-np.abs(Z2))))
    p = 1.0 / (1.0 + np.exp(-Z2))   # 这里的 Sigmoid 只用来算准确率，不参与反向

    # ---------------- 反向传播（链式法则，逐层往回推）----------------
    # 关键：dL/dZ2 = sigmoid(Z2) - y  —— Sigmoid 与 BCE 组合后的漂亮结果
    dZ2 = (p - y) / n                  # (n, 1)
    dW2 = dZ2.T @ A1                   # (out, H)
    db2 = dZ2.sum(axis=0, keepdims=True)

    dA1 = dZ2 @ W2                     # (n, H)  梯度穿过 W2 传回隐藏层
    dZ1 = dA1 * (Z1 > 0)               # ReLU 的导数：正区间 1，负区间 0
    dW1 = dZ1.T @ X                    # (H, in)
    db1 = dZ1.sum(axis=0, keepdims=True)

    # ---------------- 梯度下降更新 ----------------
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 500 == 0:
        acc = ((p > 0.5) == (y > 0.5)).mean()
        print(f"epoch {epoch:5d}  loss={loss:.6f}  acc={acc:.3f}")

Z1 = X @ W1.T + b1
A1 = np.maximum(Z1, 0)
Z2 = A1 @ W2.T + b2
p = 1.0 / (1.0 + np.exp(-Z2))
print("\n最终预测（logits 与概率）：")
for xi, pi, zi in zip(X, p.ravel(), Z2.ravel()):
    print(f"  x={xi}  logit={zi:+.3f}  p={pi:.4f}")


# ----------------------------------------------------------------------
# 3. 对照 PyTorch 写法（同样的结构，只是反向传播交给 autograd）
# ----------------------------------------------------------------------
# import torch, torch.nn as nn
#
# model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
#
# for epoch in range(2001):
#     optimizer.zero_grad()
#     logits = model(torch.tensor(X, dtype=torch.float32))
#     loss = criterion(logits, torch.tensor(y, dtype=torch.float32))
#     loss.backward()        # 这一步做的事，就是上面手写的 dW1/db1/dW2/db2
#     optimizer.step()       # 这一步做的事，就是上面手写的 -= lr * d
#
# 建议：把上面两段并排跑，逐 epoch 比对 loss 是否一致（需同样的初始化）。
# 能对上，说明你真的理解了 backward() 和 step() 在干什么。
