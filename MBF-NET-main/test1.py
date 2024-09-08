import numpy as np
import torch
from matplotlib import pyplot as plt
import albumentations as A
from Ours.Base_transformer import MBF
def norm01(x):
    return np.clip(x, 0, 255) / 255
image_data = np.load(r'')
p = 0.5
transf = A.Compose([
    A.GaussNoise(p=p),
    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),
    A.ShiftScaleRotate(p=p),
    # A.RandomBrightnessContrast(p=p),
])
tsf = transf(image=image_data.astype('uint8'))
image_data= tsf['image']
image_data = norm01(image_data)
image_data = torch.from_numpy(image_data).float()
image_data = image_data.permute(2, 0, 1)
image_data=torch.reshape(image_data,(1,3,512,512))
model = BAT(1, 50, 1, 6)
model.load_state_dict(torch.load(r''))
model.eval()
with torch.no_grad():
    output, _ = model(image_data)
    output = torch.sigmoid(output)
    output = output.cpu().numpy() > 0.6

print(output.shape)
print(output)
plt.imshow(output.squeeze(0).squeeze(0))
plt.show()


# import numpy as np
# import torch
# from matplotlib import pyplot as plt
# from Ours.Base_transformer import BAT
#
# # def norm01(x):
# #     return x / 255.0
#
# image_path = r'F:\BA-Transformer-main\data\isic2018\Image\0013585.npy'
# image = np.load(image_path)
# image = norm01(image)
# image_tensor = torch.from_numpy(image).float()
# image_tensor = torch.reshape(image_tensor, (1, 3, 512, 512))
# print(image_tensor.shape)
# model = BAT(1, 50, 1, 6)
# model.load_state_dict(torch.load(r'F:\BA-Transformer-main\logs\isic2018\_1_1_0_e6_loss_0_aug_1\fold_0\model\latest.pkl'))
# model.eval()
# with torch.no_grad():
#     output = model(image_tensor)
#     pre, map = output
#
# pre = torch.sigmoid(pre)
# plt.imshow(pre.squeeze(0).squeeze(0))
# print(pre)
# print(pre.shape)
# plt.show()