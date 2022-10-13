import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.pytorch.functional import img_to_tensor


class RoboticsDataset(Dataset):
    def __init__(self, file_names):
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path).replace('frames', 'ground_truth'), 0)

    return (mask / prepare_data.binary_factor).astype(np.uint8)
