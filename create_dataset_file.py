import os
import numpy as np
import pandas as pd
from tqdm import tqdm

data_folder = "Data/train/"

data_train = []

for file_name in tqdm(os.listdir(data_folder)):
    image_path = data_folder + file_name
    label = 0
    if "dog" in image_path:
        label = 1
    data_image = [image_path, label]
    data_train.append(data_image)

df = pd.DataFrame(data=data_train)
df.to_csv('train.csv', index=None, header=None)
