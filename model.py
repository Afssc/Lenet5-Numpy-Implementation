# %%
import numpy as np
import tqdm
import pickle
import os
import json
from abc import ABC,abstractmethod
from scipy.signal import correlate2d
from numpy.lib import stride_tricks
from matplotlib import pyplot as plt
from RBF_BITMAP import rbf_bitmap
from image_feeder import read_images_from_ubyte,read_labels_from_ubyte



# %%
class activation(ABC):
    @abstractmethod
    def gradient(self,x)->np.ndarray:
        ...
    @abstractmethod
    def __call__(self,x)->np.ndarray:
        ...
        
class layer(ABC):
    @abstractmethod
    def forward(self,vector_input:np.ndarray)->np.ndarray:
        ...
    @abstractmethod
    def backward(self,grad_input:np.ndarray,learning_rate:float)->np.ndarray:
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
        
class Lecun_Tanh(activation):
    def __call__(self, x:np.ndarray|np.matrix):
        return 1.7159 * np.tanh(2/3 * x)

    def gradient(self, x:np.ndarray|np.matrix):
        return 1.14393 * (1 - np.power(np.tanh(2/3 * x), 2))

# %%
class Fc_layer(layer):
    m_activation_func:activation
    m_batch_size=1
    m_W:np.ndarray
    m_b:np.ndarray
    _forward_in:np.ndarray
    _forward_Z:np.ndarray #线性层输出
    _forward_A:np.ndarray #激活层输出
    m_Wshape:tuple[int,int]
    # shape(in_dim,out_dim)


    def __init__(self,in_shape:tuple[int,int,int],out_shape:tuple[int,int,int],activation_func:activation=Lecun_Tanh()):
        # shape(in_dim,out_dim)
        # TODO:改batchsize
        assert len(in_shape)==3 and len(out_shape)==3  #全部改成CNN的shape输入方式
        assert in_shape[1]==out_shape[1]==1   
        assert in_shape[2]==out_shape[2]==1   
        in_dim,_,_ = in_shape
        out_dim,_,_ = out_shape
        shape = (in_dim,out_dim)
        self.m_activation_func=activation_func
        # 立昆激活函数专用初始化
        if isinstance(activation_func,Lecun_Tanh):
            normal_scale = 1/np.sqrt(in_dim)
        else:
            normal_scale = 0.01
        self.m_W = np.random.normal(loc=0.0, scale=normal_scale, size=shape)
        self.m_b = np.random.normal(loc=0.0, scale=normal_scale, size=(1, out_dim))
        self._forward_in = np.zeros((1,in_dim))
        self._forward_Z = np.zeros_like(self.m_b)
        self._forward_A = np.zeros_like(self.m_b)
        self.m_Wshape=shape
        
    

    def forward(self,vector_input:np.ndarray)->np.ndarray:
        in_dim, out_dim = self.m_Wshape
        assert np.shape(vector_input)==(in_dim,1,1) #一定要是维度匹配的行向量
        vector_input = vector_input.reshape((1,in_dim))
        self._forward_in = vector_input
        self._forward_Z = np.matmul(vector_input,self.m_W)+ self.m_b
        self._forward_A = self.m_activation_func(self._forward_Z)
        vector_output = self._forward_A.reshape((out_dim,1,1))
        return vector_output

    def backward(self,grad_input:np.ndarray,learning_rate:float=0.01) ->np.ndarray:
        in_dim, out_dim = self.m_Wshape
        assert np.shape(grad_input)==(out_dim,1,1) #一定要是维度匹配的行向量
        grad_input = grad_input.reshape((1,out_dim))
        dZ = grad_input * self.m_activation_func.gradient(self._forward_Z)
        dW = np.matmul(self._forward_in.T,dZ) / self.m_batch_size
        db = np.sum(dZ,axis=0,keepdims=True)/ self.m_batch_size
        grad_out = np.matmul(dZ,self.m_W.T).reshape((in_dim,1,1))
        self.m_W -= learning_rate * dW
        self.m_b -= learning_rate * db
        return grad_out



# %%
class Rbf_layer(layer):
    input_dim = 84
    bitmap_list:np.ndarray
    _forward_in:np.ndarray
    _forward_out:np.ndarray
    def __init__(self):
        self.bitmap_list = rbf_bitmap()  #已经导入了10个rbf位图
        self._forward_in = np.zeros((1,self.input_dim))
        self._forward_out = np.zeros((1,len(self.bitmap_list)))

        
    def forward(self,vector_input:np.ndarray)->np.ndarray:
        assert vector_input.shape == (self.input_dim,1,1)
        vector_input = vector_input.reshape((1,self.input_dim))
        self._forward_in = vector_input
        # vector_input (1,84), bitmap_list (10,84)
        diff = vector_input - self.bitmap_list  #广播,爽！
        distances = np.sum(np.square(diff), axis=1)
        return np.array(distances)

    def backward(self, grad_input:np.ndarray, learning_rate: float) :
        # 为了满足该死的接口而已,grad_input输入正确的rbf参数向量标签
        assert grad_input.shape == (1, 1, 1)
        index = int(grad_input[0, 0, 0])
        return 2*(self._forward_in - self.bitmap_list[index]).reshape((self.input_dim,1,1))
         
# %%
class Conv_layer(layer):
    m_kernel:np.ndarray
    m_b:np.ndarray
    m_in_shape:tuple  #(Channel_in,Height_in,Width_in)
    m_out_shape:tuple   #(Channel_out,Height_out,Width_out)
    m_kernel_shape:tuple  #(Valid_connection_number,Kernel_Height,Kernel_Width)
    m_bias_shape:tuple  #(1,Channel_out)
    m_connection_table:np.ndarray
    m_activation_func:activation
    m_connections:list[tuple[int,int,int]]  #(in_channel,out_channel,index)
    _forward_in:np.ndarray
    _forward_Z:np.ndarray

    def __init__(self,in_shape:tuple,out_shape:tuple,kernel_size:int,connection_table=None,activation_func:activation=Lecun_Tanh()):
        # 暂不支持batch size
        assert len(in_shape)==3
        assert len(out_shape)==3
        assert connection_table is None or connection_table.shape==(in_shape[0],out_shape[0])

        self.m_in_shape=in_shape
        self.m_out_shape=out_shape
        self.m_bias_shape = (1,out_shape[0])

        # 设置连接表
        if connection_table is None:
            # 全连接
            self.m_connection_table = np.ones((in_shape[0],out_shape[0]))
        else:
            self.m_connection_table = connection_table

        connection_pairs = np.argwhere(self.m_connection_table!=0)
        # 所有有效连接分配一个index给kernel, 拓展成3元组 (in_channel,out_channel,index)
        self.m_connections = [(in_c,out_c,idx) for idx,(in_c,out_c) in enumerate(connection_pairs)]
        

        valid_connection_num = np.count_nonzero(self.m_connection_table)
        self.m_kernel_shape = (valid_connection_num,kernel_size,kernel_size)

        if isinstance(activation_func, Lecun_Tanh): # Lecun激活函数建议的初始化，参考论文
            normal_scale = 1/np.sqrt(kernel_size*kernel_size*in_shape[0]/valid_connection_num)
        else:
            normal_scale = 0.01

        self.m_kernel = np.random.normal(loc=0.0, scale=normal_scale, size=self.m_kernel_shape)
        self.m_b = np.random.normal(loc=0.0, scale=normal_scale, size=self.m_bias_shape)

        self.m_activation_func = activation_func
        self._forward_in = np.zeros(self.m_in_shape)
        
    def forward(self,vector_input:np.ndarray)->np.ndarray:
        assert vector_input.shape == self.m_in_shape

        self._forward_in = vector_input
        output_maps = np.zeros(self.m_out_shape)
        for connection in self.m_connections:
            in_c, out_c, k_idx = connection
            # 对应输入通道做卷积
            input_map = vector_input[in_c]
            kernel = self.m_kernel[k_idx]
            conv_result = correlate2d(input_map, kernel, mode='valid')
            output_maps[out_c] += conv_result

        output_maps += self.m_b.reshape(-1,1,1)  # 广播加偏置
        self._forward_Z = output_maps   # 保存线性输出
        # 激活
        activated_maps = self.m_activation_func(output_maps)
        return activated_maps

    def backward(self,grad_input:np.ndarray,learning_rate:float)->np.ndarray: 
        # dY = grad_input * 激活函数在forward输出上的导数
        # grad_out = pad(dY) 互相关 翻转后的K  
        # dK = X 互相关 dY  
        # db = sum(grad_map_in)
        assert grad_input.shape == self.m_out_shape
        dY = grad_input * self.m_activation_func.gradient(self._forward_Z)
        dB = np.sum(dY, axis=(1,2)) / 1  # 假设batch size=1
        dKernels = np.zeros_like(self.m_kernel)
        grad_out_maps = np.zeros(self.m_in_shape)
        for connection in self.m_connections:
            in_c, out_c, k_idx = connection
            # 计算dK
            input_map = self._forward_in[in_c]
            grad_map_in = grad_input[out_c]
            dK = correlate2d(input_map, dY[out_c], mode='valid')
            dKernels[k_idx] = dK / 1  # 假设batch size=1

            # 计算grad_out
            kernel = self.m_kernel[k_idx]
            flipped_kernel = np.flip(kernel,axis=(0,1))  # 翻转kernel
            pad_height = (kernel.shape[0] - 1) 
            pad_width = (kernel.shape[1] - 1) 
            padded_grad_map = np.pad(dY[out_c], ((pad_height, pad_height), (pad_width, pad_width)), 'constant', constant_values=0)
            grad_out_map = correlate2d(padded_grad_map, flipped_kernel, mode='valid') 
            grad_out_maps[in_c] += grad_out_map
        # 更新参数
        self.m_kernel -= learning_rate * dKernels
        self.m_b -= learning_rate * dB
        return grad_out_maps

# %%
class Pooling_layer(layer):
    m_pool_size:int
    m_stride:int
    m_in_shape: tuple # (Channel_in, Height_in, Width_in)
    m_out_shape: tuple# (Channel_out, Height_out, Width_out)（Channel_out=Channel_in）
    _forward_in: np.ndarray
    def __init__(self, in_shape:tuple, pool_size:int=2, stride:int=2):
        assert len(in_shape)==3
        self.m_in_shape = in_shape
        self.m_pool_size = pool_size
        self.m_stride = stride
        channel_in, height_in, width_in = in_shape
        height_out = (height_in - pool_size) // stride + 1
        width_out = (width_in - pool_size) // stride + 1
        self.m_out_shape = (channel_in, height_out, width_out)
        self._forward_in = np.zeros(self.m_in_shape)

    def _window_view(self, x: np.ndarray) -> np.ndarray:
        # 使用stride_tricks构造滑动窗口视图 (C, H_out, W_out, P, P)
        channel_in, height_in, width_in = x.shape
        pool = self.m_pool_size
        stride = self.m_stride
        height_out = (height_in - pool) // stride + 1
        width_out = (width_in - pool) // stride + 1
        sC, sH, sW = x.strides
        return stride_tricks.as_strided(
            x,
            shape=(channel_in, height_out, width_out, pool, pool),
            strides=(sC, sH * stride, sW * stride, sH, sW),
            writeable=False,
        )

    def forward(self, vector_input: np.ndarray) -> np.ndarray:
        assert vector_input.shape == self.m_in_shape
        self._forward_in = vector_input
        windows = self._window_view(vector_input)
        return windows.mean(axis=(3, 4))
 
    def backward(self, grad_input: np.ndarray, learning_rate: float) -> np.ndarray:
        assert grad_input.shape == self.m_out_shape
        pool = self.m_pool_size
        stride = self.m_stride
        grad_out_maps = np.zeros(self.m_in_shape, dtype=grad_input.dtype)
        scale = grad_input / (pool * pool)
        if stride == pool:
            # 非重叠窗口，安全写入
            windows = stride_tricks.as_strided(
            grad_out_maps,
            shape=(self.m_in_shape[0], self.m_out_shape[1], self.m_out_shape[2], pool, pool),
            strides=(
                grad_out_maps.strides[0],
                grad_out_maps.strides[1] * stride,
                grad_out_maps.strides[2] * stride,
                grad_out_maps.strides[1],
                grad_out_maps.strides[2],
            ),
            writeable=True,
            )
            windows += scale[..., None, None]
            return grad_out_maps

        # 有重叠时使用安全累加（仍保持接口一致）
        channel_in, height_out, width_out = grad_input.shape
        row_idx = (np.arange(height_out) * stride).reshape(height_out, 1)
        col_idx = (np.arange(width_out) * stride).reshape(1, width_out)
        for ph in range(pool):
            for pw in range(pool):
                np.add.at(
                grad_out_maps,
                (
                    np.arange(channel_in)[:, None, None],
                    row_idx[None, :, :] + ph,
                    col_idx[None, :, :] + pw,
                ),
                scale[None, :, :],
                )
        return grad_out_maps




# %%

# 正式训练 + 测试（单样本训练，带tqdm）



class Lenet5(object):
    model:list[layer]
    def __init__(self):
        connection_table = np.array([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
                                    [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],
                                    [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
                                    [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],
                                    [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
                                    [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]])
        # connection_table = None
        C1 = Conv_layer((1,32,32),(6,28,28),kernel_size=5)
        S2 = Pooling_layer((6,28,28),pool_size=2,stride=2)
        C3 = Conv_layer((6,14,14),(16,10,10),kernel_size=5,connection_table=connection_table)
        S4 = Pooling_layer((16,10,10),pool_size=2,stride=2)
        C5 = Conv_layer((16,5,5),(120,1,1),kernel_size=5)
        f6 = Fc_layer((120,1,1),(84,1,1))
        output_layer = Rbf_layer()
        self.model = [C1,S2,C3,S4,C5,f6,output_layer]
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.passed_epochs = 1
        self.best_acc = -np.inf

    def forward_pass(self, image: np.ndarray) -> np.ndarray:
        x = np.array([image])  # (1, H, W)
        for layer in self.model:
            x = layer.forward(x)
        return x


    def predict_label(self, image: np.ndarray) -> int:
        distances =self.forward_pass(image)
        return int(np.argmin(distances))


    def train_one_epoch(self, images: np.ndarray, labels: np.ndarray|list[int], lr: float = 0.01, max_train: int | None = None, ui_refresh_every: int | None = None):
        correct = 0
        total = 0
        epoch_loss = 0.0
        n = images.shape[0] if max_train is None else min(images.shape[0], max_train)
        for idx in tqdm.tqdm(range(n), desc="train", leave=False):
            image = images[idx]
            label = int(labels[idx])
            # 前向
            distances = self.forward_pass(image)
            loss = float(distances[label])
            epoch_loss += loss
            # 预测
            pred = int(np.argmin(distances))
            correct += int(pred == label)
            total += 1
            # 反向（RBF层只接受标签索引）
            grad = np.array([label]).reshape((1, 1, 1))
            for layer in reversed(self.model):
                grad = layer.backward(grad, learning_rate=lr)
            if ui_refresh_every is not None and ui_refresh_every > 0 and (idx + 1) % ui_refresh_every == 0:
                plt.pause(0.001)
        return epoch_loss / max(total, 1), correct / max(total, 1)


    def evaluate(self, images: np.ndarray, labels: np.ndarray|list[int], max_test: int | None = None):
        correct = 0
        total = 0
        n = images.shape[0] if max_test is None else min(images.shape[0], max_test)
        for idx in tqdm.tqdm(range(n), desc="test", leave=False):
            image = images[idx]
            label = int(labels[idx])
            pred = self.predict_label(image)
            correct += int(pred == label)
            total += 1
        return correct / max(total, 1)

    def create_plot(self):
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1.set_ylabel("loss")

        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy (%)")

        fig.tight_layout()
        plt.show()
        return fig, ax1, ax2

    def plot_metrics(self, fig, ax1, ax2, losses, train_accs, test_accs):
        # clear_output(wait=True)
        ax1.cla()
        ax2.cla()
        ax1.plot(losses, label="loss")
        ax1.set_ylabel("loss")
        ax1.legend()

        ax2.plot([v * 100 for v in train_accs], label="train acc (%)")
        ax2.plot([v * 100 for v in test_accs], label="test acc (%)")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy (%)")
        ax2.legend()

        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)

    def train(self,train_params:dict,train_images:np.ndarray,label_list:list[int],
              test_images:np.ndarray,test_label_list:list[int]):
        params = train_params or {}
        epochs = params.get("epochs", 50)
        lr = params.get("lr", 0.0002)
        max_train = params.get("max_train", 50)  # 先小规模跑通，改成None可全量训练
        max_test = params.get("max_test", 100)
        ui_refresh_every = params.get("ui_refresh_every", 50)
        best_model_path = params.get("best_model_path", "./train/lenet5_best.pkl")
        current_model_path = params.get("current_model_path", "./train/lenet5_current.pkl")
        best_acc = self.best_acc
        fig, ax1, ax2 = self.create_plot()

        for epoch in range(self.passed_epochs, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(
                train_images,
                label_list,
                lr=lr,
                max_train=max_train,
                ui_refresh_every=ui_refresh_every,
            )
            test_acc = self.evaluate( test_images, test_label_list, max_test=max_test)
            self.loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.test_acc_history.append(test_acc)
            self.plot_metrics(fig, ax1, ax2, self.loss_history, self.train_acc_history, self.test_acc_history)
            print(f"Epoch {epoch}/{epochs} | loss: {train_loss:.4f} | train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
            self.save_model(current_model_path)
            self.passed_epochs = epoch + 1
            if test_acc > best_acc:
                best_acc = test_acc
                self.best_acc = best_acc
                self.save_model(best_model_path)
            
    def save_model(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(path: str) -> "Lenet5":
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model



# 训练参数加载
def load_train_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
# 训练集
    labels = read_labels_from_ubyte('./dataset/train-labels-idx1-ubyte')
    images = read_images_from_ubyte('./dataset/train-images-idx3-ubyte')
# 这些图都是28x28的，padding到32x32
    images = np.pad(images, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    images = images.astype("float32")/127.5 - 1.0
# 测试集
    test_labels = read_labels_from_ubyte('./dataset/t10k-labels-idx1-ubyte')
    test_images = read_images_from_ubyte('./dataset/t10k-images-idx3-ubyte')
# 这些图也是28x28的，padding到32x32
    test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    test_images = test_images.astype("float32")/127.5 - 1.0
     
    train_dir = "./train"
    os.makedirs(train_dir, exist_ok=True)
    train_files = [name for name in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, name))]
    current_model_path = os.path.join(train_dir, "lenet5_current.pkl")

    if train_files:
        load_path = current_model_path if os.path.exists(current_model_path) else os.path.join(train_dir, train_files[0])
        print(f"train目录已有文件，加载模型: {load_path}")
        lenet5 = Lenet5.load_model(load_path)
    else:
        print("train目录为空，已创建目录并新建模型")
        lenet5 = Lenet5()

    train_params = load_train_params("./train_params.json")
    lenet5.train(train_params, images, labels, test_images, test_labels)