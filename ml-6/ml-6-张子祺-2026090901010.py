import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化RNN参数
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier / Glorot 初始化
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(
            2.0 / (input_size + hidden_size)
        )
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(
            2.0 / (hidden_size + hidden_size)
        )
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(
            2.0 / (hidden_size + output_size)
        )

        # 偏置初始化
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        # 前向传播缓存
        self.inputs = None
        self.hidden_states = None
        self.outputs = None

        # 梯度容器
        self.grads = {}

    def forward(self, inputs):
        """
        前向传播
        inputs: shape (seq_len, batch_size, input_size)
        return: outputs, hidden_states
        """
        seq_len, batch_size, _ = inputs.shape
        self.inputs = inputs

        # h_0 初始化
        h = np.zeros((self.hidden_size, batch_size))

        # 保存所有时间步的隐藏状态
        self.hidden_states = np.zeros((seq_len + 1, self.hidden_size, batch_size))
        self.hidden_states[0] = h

        # 保存所有时间步的输出
        outputs = np.zeros((seq_len, self.output_size, batch_size))

        for t in range(seq_len):
            x_t = inputs[t].T

            h = np.tanh(self.W_xh @ x_t + self.W_hh @ h + self.b_h)
            self.hidden_states[t + 1] = h

            y_t = self.W_hy @ h + self.b_y
            outputs[t] = y_t

        self.outputs = outputs

        outputs_out = outputs.transpose(0, 2, 1)
        hidden_out = self.hidden_states[1:].transpose(0, 2, 1)  # h_1 ... h_T
        return outputs_out, hidden_out

    def backward(self, d_outputs):
        """
        反向传播 - 通过时间反向传播(BPTT)
        d_outputs: 输出层的梯度, shape (seq_len, batch_size, output_size)
        return: 各参数的梯度 dict
        """
        seq_len, batch_size, _ = self.inputs.shape

        d_outputs = d_outputs.transpose(0, 2, 1)

        # 初始化梯度
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # 来自未来时间步的隐藏状态梯度
        dh_next = np.zeros((self.hidden_size, batch_size))

        for t in reversed(range(seq_len)):
            h_t = self.hidden_states[t + 1]       # (hidden_size, batch_size)
            h_prev = self.hidden_states[t]        # (hidden_size, batch_size)
            x_t = self.inputs[t].T                # (input_size, batch_size)

            dy = d_outputs[t]                     # (output_size, batch_size)

            # 输出层梯度
            dW_hy += dy @ h_t.T
            db_y += dy.sum(axis=1, keepdims=True)

            # 隐藏状态梯度
            dh = self.W_hy.T @ dy + dh_next

            dh_raw = dh * (1.0 - h_t ** 2)

            # 参数梯度
            dW_xh += dh_raw @ x_t.T
            dW_hh += dh_raw @ h_prev.T
            db_h += dh_raw.sum(axis=1, keepdims=True)

            # 传递给上一个时间步
            dh_next = self.W_hh.T @ dh_raw

        self.grads = {
            'W_xh': dW_xh,
            'W_hh': dW_hh,
            'W_hy': dW_hy,
            'b_h': db_h,
            'b_y': db_y,
        }
        return self.grads

    def update_parameters(self, learning_rate):
        """
        参数更新（SGD）
        """
        for name in ('W_xh', 'W_hh', 'W_hy', 'b_h', 'b_y'):
            setattr(self, name, getattr(self, name) - learning_rate * self.grads[name])

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        """
        LSTM单元初始化
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = hidden_size + input_size
        self.concat_size = concat_size

        fan_in = concat_size
        bound = 1.0 / np.sqrt(fan_in)

        # 遗忘门
        self.W_f = np.random.uniform(-bound, bound, (hidden_size, fan_in))
        self.b_f = np.zeros(hidden_size)

        # 输入门
        self.W_i = np.random.uniform(-bound, bound, (hidden_size, fan_in))
        self.b_i = np.zeros(hidden_size)

        # 候选记忆
        self.W_g = np.random.uniform(-bound, bound, (hidden_size, fan_in))
        self.b_g = np.zeros(hidden_size)

        # 输出门
        self.W_o = np.random.uniform(-bound, bound, (hidden_size, fan_in))
        self.b_o = np.zeros(hidden_size)

        self.cache = None

    def forward(self, input_t, hidden_state, cell_state):
        """
        LSTM单元前向传播
        input_t:      [batch_size, input_size]
        hidden_state: [batch_size, hidden_size]
        cell_state:   [batch_size, hidden_size]
        return: new_hidden_state, new_cell_state
        """
        h_prev = hidden_state
        c_prev = cell_state

        x_cat = np.concatenate([h_prev, input_t], axis=1)

        f = _sigmoid(x_cat @ self.W_f.T + self.b_f)
        i = _sigmoid(x_cat @ self.W_i.T + self.b_i)
        o = _sigmoid(x_cat @ self.W_o.T + self.b_o)
        g = np.tanh(x_cat @ self.W_g.T + self.b_g)

        c_new = f * c_prev + i * g

        h_new = o * np.tanh(c_new)

        self.cache = (x_cat, f, i, g, o, c_prev, c_new)

        return h_new, c_new

def generate_sequence_data(seq_type='fibonacci', length=1000):
    """
    生成序列数据。

    参数
    ----
    seq_type : {'fibonacci', 'sine', 'linear'}
    length   : 序列长度

    返回
    ----
    seq : np.ndarray, shape (length,)
        原始（未归一化）序列。
    """
    if length < 2:
        raise ValueError("length 至少为 2")

    if seq_type == 'fibonacci':
        fib = [0, 1]
        for _ in range(length - 2):
            fib.append(fib[-1] + fib[-2])
        seq = np.array(fib[:length], dtype=np.float64)
        return seq

    elif seq_type == 'sine':
        t = np.arange(length, dtype=np.float64)
        return np.sin(2 * np.pi * t / 25.0) + 0.5 * np.sin(2 * np.pi * t / 7.0)

    elif seq_type == 'linear':
        return np.arange(length, dtype=np.float64)

# ---------------------------------------------------------------------------
# 2. 训练 / 评估 / 可视化
# ---------------------------------------------------------------------------
def train_rnn_on_sequence():
    """
    训练 SimpleRNN 进行斐波那契序列的一步预测。

    流程：
        1) 生成并归一化序列数据
        2) 用滑动窗口构造 (window, 1) -> next 的样本
        3) 按时间顺序划分训练 / 测试集
        4) 训练 RNN（MSE + SGD + 梯度裁剪）
        5) 评估 & 可视化
    """
    seq = generate_sequence_data('fibonacci', length=200)

    seq_min = float(seq.min())
    seq_max = float(seq.max())
    scale = seq_max - seq_min + 1e-12
    seq_norm = (seq - seq_min) / scale

    window = 10
    X_list, Y_list = [], []
    for i in range(len(seq_norm) - window):
        X_list.append(seq_norm[i:i + window])
        Y_list.append(seq_norm[i + window])
    X = np.asarray(X_list, dtype=np.float32).reshape(-1, window, 1)   # (N, T, 1)
    Y = np.asarray(Y_list, dtype=np.float32).reshape(-1, 1)           # (N, 1)

    split = int(len(X) * 0.8)
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]
    print(f"[数据] 总样本 {len(X)} | 训练 {len(X_tr)} | 测试 {len(X_te)} | window={window}")

    hidden_size = 32
    model = SimpleRNN(input_size=1, hidden_size=hidden_size, output_size=1)

    epochs = 150
    batch_size = 16
    lr = 1e-2
    max_grad_norm = 5.0
    losses = []

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_tr))
        epoch_loss = 0.0

        for start in range(0, len(X_tr), batch_size):
            idx = perm[start:start + batch_size]
            xb = X_tr[idx].transpose(1, 0, 2)
            yb = Y_tr[idx]

            outputs, _ = model.forward(xb)
            pred = outputs[-1]

            diff = pred - yb
            mse = float(np.mean(diff ** 2))
            epoch_loss += mse * len(idx)

            d_pred = 2.0 * diff / len(idx)
            d_outputs = np.zeros_like(outputs)
            d_outputs[-1] = d_pred

            grads = model.backward(d_outputs)

            total_norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            if total_norm > max_grad_norm:
                k = max_grad_norm / (total_norm + 1e-8)
                for name in grads:
                    grads[name] *= k

            model.update_parameters(lr)

        avg_loss = epoch_loss / len(X_tr)
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            lr *= 0.5
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[训练] Epoch {epoch + 1:3d}/{epochs} | MSE = {avg_loss:.3e} | lr = {lr:.2e}")

    def _predict(model, X_in, bs=64):
        preds = []
        for s in range(0, len(X_in), bs):
            xb = X_in[s:s + bs].transpose(1, 0, 2)  # (T, B, 1)
            out, _ = model.forward(xb)
            preds.append(out[-1])                   # (B, 1)
        return np.concatenate(preds, axis=0)

    pred_te_norm = _predict(model, X_te)             # (N_te, 1)
    rmse_norm = float(np.sqrt(np.mean((pred_te_norm - Y_te) ** 2)))

    pred_te = pred_te_norm * scale + seq_min
    true_te = Y_te * scale + seq_min
    rmse_raw = float(np.sqrt(np.mean((pred_te - true_te) ** 2)))

    print("\n========== 评估结果 ==========")
    print(f"测试集 RMSE (归一化空间) : {rmse_norm:.3e}")
    print(f"测试集 RMSE (原始尺度)   : {rmse_raw:.3e}")
    print(f"测试集最后一项真实值     : {true_te[-1, 0]:.6e}")
    print(f"测试集最后一项预测值     : {pred_te[-1, 0]:.6e}")
    print("\n----- 部分预测示例 (原始尺度) -----")
    for i in [0, len(true_te) // 4, len(true_te) // 2, 3 * len(true_te) // 4, -1]:
        print(f"  真实: {true_te[i, 0]:>15.6e} | 预测: {pred_te[i, 0]:>15.6e}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        axes[0].plot(losses, color='#1f77b4', lw=1.5)
        axes[0].set_title('Training MSE (normalized)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        show = min(100, len(true_te))
        axes[1].plot(Y_te[:show], label='True (normalized)',
                     color='#2ca02c', lw=1.8)
        axes[1].plot(pred_te_norm[:show], label='Prediction',
                     color='#d62728', lw=1.4, linestyle='--')
        axes[1].set_title(f'Test set (first {show} points) | RMSE={rmse_norm:.2e}')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Normalized value')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = 'rnn_fibonacci_prediction.png'
        plt.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"\n[可视化] 图像已保存到: {save_path}")

    except Exception as e:
        print(f"\n[警告] 可视化失败（可能未安装 matplotlib）：{e}")

    return {
        'model': model,
        'losses': losses,
        'predictions': pred_te,
        'targets': true_te,
        'rmse_normalized': rmse_norm,
        'rmse_raw': rmse_raw,
        'normalization': {'min': seq_min, 'max': seq_max},
    }

if __name__ == '__main__':
    results = train_rnn_on_sequence()

#  以上代码由本人手搓后交由deepseek检查改正，并增加可用性
#  可视化部分全部由deepseek生成