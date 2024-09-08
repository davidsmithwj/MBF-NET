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
model = MBF(1, 50, 1, 6)
model.load_state_dict(torch.load(r''))
model.eval()
with torch.no_grad():
    output, _ = model(image_data)
    output = torch.sigmoid(output)


    output = output.cpu().numpy() > 0.5
    # print(np.unique(output))

print(output.shape)
print(output)
plt.imshow(output.squeeze(0).squeeze(0))
plt.show()