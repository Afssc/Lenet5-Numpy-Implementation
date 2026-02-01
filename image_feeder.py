# 载入不同种类的dataset
import numpy as np
import struct
import matplotlib.pyplot as plt


# 从网上抄来的
def read_images_from_ubyte(file_path):
    with open(file_path, 'rb') as file:
        # 读取魔数和图像信息
        magic_number, num_images, num_rows, num_columns = struct.unpack('>IIII', file.read(16))
        print(f"Magic Number: {magic_number}, Number of Images: {num_images}, Rows: {num_rows}, Columns: {num_columns}")
        
        # 读取所有图像
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, num_rows, num_columns)
    
    return images


def read_labels_from_ubyte(file_path) -> list[int]:
    with open(file_path, 'rb') as file:
        # 读取魔数和标签数量
        magic_number, num_labels = struct.unpack('>II', file.read(8))
        print(f"Magic Number: {magic_number}, Number of Labels: {num_labels}")
        
        # 读取所有标签
        labels = []
        for _ in range(num_labels):
            label = struct.unpack('B', file.read(1))[0]
            labels.append(label)
    
    return labels

if "__main__" == __name__:

    file_path = './dataset/train-labels-idx1-ubyte'
    labels = read_labels_from_ubyte(file_path)
    print(f"First 10 Labels: {labels[:10]}")

    file_path = './dataset/train-images-idx3-ubyte'
    images = read_images_from_ubyte(file_path)
    print(f"First Image Shape: {images[0].shape}")

# 显示第一张图像
    plt.figure()
    plt.imshow(images[0], cmap='gray')
    plt.show()