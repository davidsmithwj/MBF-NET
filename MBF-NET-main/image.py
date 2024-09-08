import numpy as np

# 读取.npy文件
image_data = np.load(r'F:\BA-Transformer-main\data\isic2018\Image\0000000.npy')  # 将'your_image.npy'替换为你要读取的图像文件名

# 获取图像形状
image_shape = image_data.shape
print('图像形状:', image_shape)

# 获取图像数据类型
image_dtype = image_data.dtype
print('图像数据类型:', image_dtype)