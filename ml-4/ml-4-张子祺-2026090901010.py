import numpy as np
from numpy.lib.stride_tricks import as_strided


def _pair(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, init='kaiming'):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = _pair(kernel_size)
        self.stride       = _pair(stride)
        self.padding      = _pair(padding)
        self.dilation     = _pair(dilation)
        self.use_bias     = bias

        kh, kw = self.kernel_size
        fan_in  = in_channels  * kh * kw
        fan_out = out_channels * kh * kw

        # ---------- 权重初始化 ----------
        if   init == 'kaiming': std = np.sqrt(2.0 / fan_in)
        elif init == 'xavier' : std = np.sqrt(2.0 / (fan_in + fan_out))
        else                  : std = np.sqrt(1.0 / fan_in)
        self.weight      = (np.random.randn(out_channels, in_channels, kh, kw) * std).astype(np.float32)
        self.grad_weight = None

        # ---------- 偏置 ----------
        if bias:
            self.bias      = np.zeros(out_channels, dtype=np.float32)
            self.grad_bias = None
        else:
            self.bias = self.grad_bias = None

        self._cache = {}

    # ---------- PyTorch 风格的小工具 ----------
    def __call__(self, x):        return self.forward(x)
    @property
    def params(self):             return [self.weight] + ([self.bias] if self.use_bias else [])
    @property
    def grads(self):              return [self.grad_weight] + ([self.grad_bias] if self.use_bias else [])
    def zero_grad(self):
        self.grad_weight = None
        self.grad_bias   = None

    # ---------- 前向 ----------
    def _pad(self, x):
        ph, pw = self.padding
        if ph == 0 and pw == 0:
            return x
        return np.pad(x, ((0,0),(0,0),(ph,ph),(pw,pw)), mode='constant')

    def _im2col(self, x):
        """x: (N, C, H, W) 已 padding。返回 cols: (N*out_h*out_w, C*kh*kw)"""
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        out_h = (H - dh*(kh-1) - 1) // sh + 1
        out_w = (W - dw*(kw-1) - 1) // sw + 1

        sN, sC, sH, sW = x.strides

        view = as_strided(
            x,
            shape=(N, C, out_h, out_w, kh, kw),
            strides=(sN, sC, sH*sh, sW*sw, sH*dh, sW*dw),
        )

        cols = view.transpose(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)
        return cols, out_h, out_w

    def forward(self, x):
        xp = self._pad(x)
        cols, out_h, out_w = self._im2col(xp)
        N = x.shape[0]

        W2 = self.weight.reshape(self.out_channels, -1)
        out = cols @ W2.T
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        if self.use_bias:
            out = out + self.bias.reshape(1, -1, 1, 1)

        self._cache = dict(x_shape=x.shape, cols=cols, out_h=out_h, out_w=out_w)
        return out

    # ---------- 反向 ----------
    def _col2im(self, grad_cols, x_shape):
        """把 (N*out_h*out_w, C*kh*kw) 散射回 (N, C, H+2p, W+2p)"""
        N, C, H, W = x_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        ph, pw = self.padding
        out_h = (H + 2*ph - dh*(kh-1) - 1) // sh + 1
        out_w = (W + 2*pw - dw*(kw-1) - 1) // sw + 1

        g = grad_cols.reshape(N, out_h, out_w, C, kh, kw)
        g = g.transpose(0, 3, 4, 5, 1, 2)

        gx = np.zeros((N, C, H + 2*ph, W + 2*pw), dtype=g.dtype)
        for i in range(kh):
            i_end = i*dh + (out_h-1)*sh + 1
            for j in range(kw):
                j_end = j*dw + (out_w-1)*sw + 1
                gx[:, :, i*dh:i_end:sh, j*dw:j_end:sw] += g[:, :, i, j]
        if ph or pw:
            gx = gx[:, :, ph:ph+H, pw:pw+W]
        return gx

    def backward(self, grad_out):
        c = self._cache
        N = c['x_shape'][0]
        cols = c['cols']

        g = grad_out.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)


        self.grad_weight = (g.T @ cols).reshape(self.weight.shape)

        if self.use_bias:
            self.grad_bias = g.sum(axis=0)


        W2 = self.weight.reshape(self.out_channels, -1)
        grad_cols = g @ W2
        return self._col2im(grad_cols, c['x_shape'])

    def _pair(v):
        """kernel/stride/padding 统一成 (h, w)"""
        if isinstance(v, (tuple, list)):
            assert len(v) == 2
            return (int(v[0]), int(v[1]))
        return (int(v), int(v))

    def _sliding_windows(x, kernel_size, stride):
        """
        x: (N, C, H, W) 已经 padding 过
        返回零拷贝视图: (N, C, Ho, Wo, kh, kw)
        """
        N, C, H, W = x.shape
        kh, kw = kernel_size
        sh, sw = stride
        Ho = (H - kh) // sh + 1
        Wo = (W - kw) // sw + 1
        s0, s1, s2, s3 = x.strides
        return as_strided(
            x,
            shape=(N, C, Ho, Wo, kh, kw),
            strides=(s0, s1, s2 * sh, s3 * sw, s2, s3),
            writeable=False,
        )

def _sliding_windows(x, kernel_size, stride):
    """
    x: (N, C, H, W) 已经 padding 过
    返回零拷贝视图: (N, C, Ho, Wo, kh, kw)
    """
    N, C, H, W = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    Ho = (H - kh) // sh + 1
    Wo = (W - kw) // sw + 1
    s0, s1, s2, s3 = x.strides
    return as_strided(
        x,
        shape=(N, C, Ho, Wo, kh, kw),
        strides=(s0, s1, s2 * sh, s3 * sw, s2, s3),
        writeable=False,
    )


class _Pool2dBase:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = _pair(kernel_size)

        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
        self._cache = None

    def _to_4d(self, x):
        x = np.ascontiguousarray(np.asarray(x))
        if x.ndim == 3:
            return x[None], True
        if x.ndim == 4:
            return x, False
        raise ValueError(f"expect 3D/4D input, got shape {x.shape}")

    def _pad(self, x, value=0.0):
        ph, pw = self.padding
        if ph == 0 and pw == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                      mode="constant", constant_values=value)

    def _unpad_grad(self, dxp, x_shape):
        ph, pw = self.padding
        H, W = x_shape[2], x_shape[3]
        return dxp[:, :, ph:ph + H, pw:pw + W]

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

class MaxPool2d(_Pool2dBase):
    """最大值池化"""

    def forward(self, x):
        x, squeeze = self._to_4d(x)
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)

        xp = self._pad(x, value=-np.inf)

        windows = _sliding_windows(xp, self.kernel_size, self.stride)
        out = windows.max(axis=(-2, -1))          # (N, C, Ho, Wo)

        self._cache = (windows, xp.shape, x.shape, squeeze)
        return out[0] if squeeze else out

    def backward(self, grad):
        windows, xp_shape, x_shape, squeeze = self._cache
        if squeeze:
            grad = grad[None]

        N, C, Ho, Wo, kh, kw = windows.shape

        idx = windows.argmax(axis=(-2, -1))
        ki, kj = idx // kw, idx % kw

        sh, sw = self.stride
        rows = (np.arange(Ho) * sh)[None, None, :, None] + ki   # (N,C,Ho,Wo)
        cols = (np.arange(Wo) * sw)[None, None, None, :] + kj

        ni = np.arange(N)[:, None, None, None]
        ci = np.arange(C)[None, :, None, None]

        dxp = np.zeros(xp_shape, dtype=grad.dtype)

        np.add.at(dxp, (ni, ci, rows, cols), grad)

        dx = self._unpad_grad(dxp, x_shape)
        return dx[0] if squeeze else dx

class AvgPool2d(_Pool2dBase):
    """平均池化（padding 计 0 并计入分母，等价 count_include_pad=True）"""

    def forward(self, x):
        x, squeeze = self._to_4d(x)
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)

        xp = self._pad(x, value=0.0)
        windows = _sliding_windows(xp, self.kernel_size, self.stride)
        out = windows.mean(axis=(-2, -1))

        self._cache = (xp.shape, x.shape, squeeze)
        return out[0] if squeeze else out

    def backward(self, grad):
        xp_shape, x_shape, squeeze = self._cache
        if squeeze:
            grad = grad[None]

        kh, kw = self.kernel_size
        sh, sw = self.stride
        Ho, Wo = grad.shape[2], grad.shape[3]

        dxp = np.zeros(xp_shape, dtype=grad.dtype)
        g = grad / (kh * kw)

        for i in range(kh):
            for j in range(kw):
                dxp[:, :, i:i + Ho * sh:sh, j:j + Wo * sw:sw] += g

        dx = self._unpad_grad(dxp, x_shape)
        return dx[0] if squeeze else dx