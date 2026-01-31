# %%
import numpy as np
from numpy.linalg import matmul
from abc import ABC,abstractmethod
from matplotlib import pyplot as plt
from RBF_BITMAP import rbf_bitmap


# %% [markdown]
# **正向传播**
# $$
# \left(
# \begin{matrix}
# x_{1}&x_{2}&x_{3}\\
# \end{matrix}
# \right)
# 
# \left(
# \begin{matrix}
# w_{1}&w_{4} \\
# w_{2}&w_{5} \\
# w_{3}&w_{6}
# \end{matrix}
# \right)
# +
# \left(
# \begin{matrix}
# b_{1}&b_{2}
# \end{matrix}
# \right
# )
# =
# \left(
# \begin{matrix}
# y_{1}&y_{2}
# \end{matrix}
# \right)
# $$

# %% [markdown]
# **反向传播**
# $$
# \begin{align}
# &dZ=\Delta_{in}⊙f'(Z_{i}) \\
# &dW=\frac{1}{batch} X^T \cdot dZ \\
# &db=\frac{1}{batch}\sum_{i=1}^{batch}dZ_{i}
# \end{align}
# $$
#  

# %%
class activation(ABC):
    @abstractmethod
    def gradient(self,x)->...:
        ...
    @abstractmethod
    def __call__(self,x)->...:
        ...

# %%
class Sigmoid(activation):
    def __call__(self, x:np.ndarray|np.matrix):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x:np.ndarray|np.matrix):
        return self.__call__(x) * (1 - self.__call__(x))
        
class Tanh(activation):
    def __call__(self, x:np.ndarray|np.matrix):
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradient(self, x:np.ndarray|np.matrix):
        return 1 - np.power(self.__call__(x), 2)
        
class Custom_sigmoid(activation):
    def __call__(self, x:np.ndarray|np.matrix):
        return 1.7159 * np.tanh(2/3 * x)

    def gradient(self, x:np.ndarray|np.matrix):
        return 1.14393 * (1 - np.power(np.tanh(2/3 * x), 2))

# %%
class Fc_layer(object):
    m_activation_func:activation=Custom_sigmoid()
    m_batch_size=1
    m_W:np.ndarray
    m_b:np.ndarray
    _forward_in:np.ndarray
    _forward_Z:np.ndarray #线性层输出
    _forward_A:np.ndarray #激活层输出
    m_shape:tuple
    # shape(in_dim,out_dim)


    def __init__(self,shape:tuple,batch_size:int=1):
        # shape(in_dim,out_dim)
        # TODO:改batchsize
        in_dim, out_dim = shape
        # self.m_W=np.zeros(shape) 
        # self.m_b=np.zeros((1,out_dim))
        self.m_W = np.random.normal(loc=0.0, scale=0.01, size=shape)
        self.m_b = np.random.normal(loc=0.0, scale=0.01, size=(1, out_dim))
        self._forward_in = np.zeros_like(self.m_b)
        self._forward_Z = np.zeros_like(self.m_b)
        self._forward_A = np.zeros_like(self.m_b)
        self.m_shape=shape
        self.m_batch_size=batch_size
        
    def forward(self,vec_in:np.ndarray|np.matrix)->np.ndarray|np.matrix:
        in_dim, out_dim = self.m_shape
        assert np.shape(vec_in)==(1,in_dim) #一定要是维度匹配的行向量
        self._forward_in = vec_in
        self._forward_Z = np.matmul(vec_in,self.m_W)+ self.m_b
        self._forward_A = self.m_activation_func(self._forward_Z)
        return self._forward_A

    def backward(self,grad_in:np.ndarray,lr:float=0.01):
        in_dim, out_dim = self.m_shape
        assert np.shape(grad_in)==(1,out_dim)
        dZ = grad_in * self.m_activation_func.gradient(self._forward_Z)
        dW = np.matmul(self._forward_in.T,dZ) / self.m_batch_size
        db = np.sum(dZ,axis=0,keepdims=True)/ self.m_batch_size
        grad_out = np.matmul(dZ,self.m_W.T)
        self.m_W -= lr * dW
        self.m_b -= lr * db
        return grad_out



# %%
class Rbf_layer(object):
    input_dim = 84
    bitmap_list:np.ndarray
    _forward_in:np.ndarray
    _forward_out:np.ndarray
    def __init__(self):
        self.bitmap_list = rbf_bitmap()  #已经导入了10个rbf位图
        self._forward_in = np.zeros((1,self.input_dim))
        self._forward_out = np.zeros((1,len(self.bitmap_list)))

    def calculate_euclidean_distance(self,vec_in:np.ndarray|np.matrix)->np.ndarray:
        distances = []
        for bitmap in self.bitmap_list:
            dist = np.linalg.norm(vec_in - bitmap)
            distances.append(dist)
        return np.array(distances).reshape(1, -1)

    def calculate_loss_gradient(self,correct_label):
        # 输出层的loss关于输入的梯度，输出1x84行向量
        return 2*(self._forward_in - self.bitmap_list[correct_label]).reshape(1,-1)


# %%

f6 = Fc_layer((120,84),batch_size=1)
output_layer = Rbf_layer()


# %%



