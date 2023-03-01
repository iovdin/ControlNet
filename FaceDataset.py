import json
import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

class FaceDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('dataset/dataset_with_captions.csv')

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        source_filename = f"{item['user_id']}_{item['input']}"
        target_filename = f"{item['user_id']}_{item['target']}"
        prompt = item['caption']

        source = cv2.imread('./dataset/' + source_filename)
        target = cv2.imread('./dataset/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = crop_square(source, 512)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = crop_square(target, 512)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
dataset = FaceDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)