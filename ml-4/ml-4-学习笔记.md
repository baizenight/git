## 创建卷积层代码逐行讲解
### 1.导入与辅助函数

```
import numpy as np
from numpy.lib.stride_tricks import as_strided
```
***
### 2.构造函数把标量和元组统一为二元组

```
def _pair(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)
```
其中：
```
isinstance(obj, type) # 判断一个对象是不是某个类型
```
***
### 3.构造卷积层函数

```
class Conv2d: 
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, init='kaiming'):
```
其中：
```
in_channels # 输入通道数 
out_channels # 输出通道数
kernel_size # 卷积核大小
stride # 步长
padding # 零填充
dilation # 空洞膨胀系数
bias # 是否使用偏置项
init # 权重初始化方式
```
权重初始化方式：
	`'kaiming'`：He 初始化，适合后接 ReLU 的网络
	`'xavier'`：Glorot 初始化，适合 tanh/sigmoid
***
### 4.参数保存为实例属性

```
self.in_channels  = in_channels  
self.out_channels = out_channels  
self.kernel_size  = _pair(kernel_size)  
self.stride       = _pair(stride)  
self.padding      = _pair(padding)  
self.dilation     = _pair(dilation)  
self.use_bias     = bias
```
其中：
`self.xxx`是给对象创建一个属性
***
### 5.权重初始化

```
kh, kw = self.kernel_size  
fan_in  = in_channels  * kh * kw  
fan_out = out_channels * kh * kw
```
其中：
`kh, kw = self.kernel_size`是把二元组解包成两个向量
`fan_in`是“扇入”局部窗口的大小
`fan_out`是“扇出”一个输入元素所影响的输出元素个数
如果初始化不当就会导致权重方差过大或过小，导致梯度爆炸或者梯度消失

```
if   init == 'kaiming': std = np.sqrt(2.0 / fan_in)  
elif init == 'xavier' : std = np.sqrt(2.0 / (fan_in + fan_out))  
else                  : std = np.sqrt(1.0 / fan_in)  
self.weight      = (np.random.randn(out_channels, in_channels, kh, kw) * std).astype(np.float32)  
self.grad_weight = None
```
其中：
初始化要做两件事情：打破对称性，控制方差
 `if init == 'kaiming': std = np.sqrt(2.0 / fan_in)`
 
**为什么乘 2**：ReLU 把负半轴全砍成 0，相当于**方差减半**（输出只有一半元素非零）。为了补偿，权重方差要翻倍，所以分子是 `2` 而不是 `1`

这是**现代 CNN 最常用的默认初始化**

`np.random.randn(out_channels, in_channels, kh, kw)`

生成一个**形状 `(O, C, kh, kw)`、服从标准正态分布 `N(0, 1)`** 的随机张量，每个元素独立采样，后`*std`相当于把标准正态"缩放"成目标标准差

`.astype(np.float32)`把 NumPy 默认的 `float64` 转成 `float32`

`self.grad_weight = None`占位，设置梯度为空
***
### 6.偏置初始化

```
if bias:
    self.bias      = np.zeros(out_channels, dtype=np.float32)
    self.grad_bias = None
else:
    self.bias = self.grad_bias = None

self._cache = {}
```
`if bias:`是判断实例属性的bool值

`self.bias      = np.zeros(out_channels, dtype=np.float32)`

初始化偏置向量为(out_channels, )
深度学习里的**标准约定**：偏置全部初始化为0

`self._cache = {}`
 
一个空字典，**为反向传播准备**。前向时会往里塞中间结果，反向时取出

创建为字典而不是属性，方便清空缓存
***
### 7.PyTorch 风格的辅助接口

```
def __call__(self, x):        return self.forward(x)
@property
def params(self):             return [self.weight] + ([self.bias] if self.use_bias else [])
@property
def grads(self):              return [self.grad_weight] + ([self.grad_bias] if self.use_bias else [])
def zero_grad(self):
    self.grad_weight = None
    self.grad_bias   = None
```





