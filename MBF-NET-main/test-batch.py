import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import albumentations as A
from tqdm import tqdm

from Ours.Base_transformer import MBF


def norm01(x):
    return np.clip(x, 0, 255) / 255

# Function to save output data
def save_output(output, output_dir, filename):
    # Assuming output is a binary mask
    output = output.astype(np.uint8) * 255
    plt.imsave(os.path.join(output_dir, filename), output.squeeze(), cmap='gray')

# Assuming image_dir is the directory where your images are stored
image_dir = r''
output_dir = r''  # Directory to save output data
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Assuming you have a list of filenames in your dataset
file_list = os.listdir(image_dir)  # Add your filenames here

p = 0.5
transf = A.Compose([
    A.GaussNoise(p=p),
    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),
    A.ShiftScaleRotate(p=p),
    # A.RandomBrightnessContrast(p=p),
])

# Initialize your model
model = MBF(1, 50, 1, 6)
model.load_state_dict(torch.load(r''))
model.eval()

# Iterate over your dataset
for filename in tqdm(file_list):
    # Load image data
    image_data = np.load(os.path.join(image_dir, filename))

    # Apply transformations
    tsf = transf(image=image_data.astype('uint8'))
    image_data = tsf['image']
    image_data = norm01(image_data)
    image_data = torch.from_numpy(image_data).float()
    image_data = image_data.permute(2, 0, 1)
    image_data = torch.reshape(image_data, (1, 3, 512, 512))

    # Perform inference
    with torch.no_grad():
        output, pointmap = model(image_data)
        output = torch.sigmoid(pointmap[6])
        output = output.cpu().numpy() > 0.5

    # Save output data
    save_output(output, output_dir, filename[:-1] + '.png')  # Assuming you want to save as PNG format