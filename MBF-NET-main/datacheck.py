import numpy as np

file_path=r""

data=np.load(file_path)
np.set_printoptions(threshold=np.inf)
print(data)