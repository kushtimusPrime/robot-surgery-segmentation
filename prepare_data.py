"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np

import albumentations as A

import warnings
warnings.filterwarnings("ignore")

data_path = Path('Dataset')

original_path = data_path / 'original_data' / 'instrument_1_4_training'

training_path = data_path / 'training_data'

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255

num_augmentation_per_image = 6

def make_directories():
    for instrument_number in range(1, 5):
        for sub1 in ['training', 'validation']:
            for sub2 in ['frames', 'ground_truth']:
                (training_path / f'instrument{instrument_number}' / sub1 / sub2).mkdir(exist_ok=True, parents=True)
    
def make_transform():
    return A.Compose([
        A.Flip(),
        A.GaussNoise(p=0.5),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.6, rotate_limit=180, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.4),
            A.GridDistortion(p=.2),
            A.PiecewiseAffine(p=0.4),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.5),
        A.HueSaturationValue(p=0.75),
    ])

if __name__ == '__main__':
    make_directories()
    
    transform = make_transform()
    
    for instrument_number in range(1, 5):
        images = list((original_path / ('instrument_dataset_' + str(instrument_number)) / 'left_frames').glob('*'))
        mask_folders = list((original_path / ('instrument_dataset_' + str(instrument_number)) / 'ground_truth').glob('*'))

        for file_name in tqdm(images):
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape

            img = img[h_start: h_start + height, w_start: w_start + width]

            mask_binary = np.zeros((old_h, old_w), dtype=np.uint8)

            for mask_folder in mask_folders:
                mask = cv2.imread(str(mask_folder / file_name.name), 0)
                if not mask is None:
                    mask_binary += mask

            mask_binary = (mask_binary[h_start: h_start + height, w_start: w_start + width] > 0).astype(
                np.uint8) * binary_factor
            
            cv2.imwrite(str(training_path / f'instrument{instrument_number}' / 'validation' / 'frames' / f'{file_name.stem}.png'), img)
            cv2.imwrite(str(training_path / f'instrument{instrument_number}' / 'validation' / 'ground_truth' / f'{file_name.stem}.png'), mask_binary)
            
            cv2.imwrite(str(training_path / f'instrument{instrument_number}' / 'training' / 'frames' / f'{file_name.stem}_0.png'), img)
            cv2.imwrite(str(training_path / f'instrument{instrument_number}' / 'training' / 'ground_truth' / f'{file_name.stem}_0.png'), mask_binary)
            
            for augmentation_count in range(num_augmentation_per_image):
                transformed = transform(image=img, mask=mask_binary)
                
                cv2.imwrite(str(training_path / f'instrument{instrument_number}' / 'training' / 'frames' / f'{file_name.stem}_{augmentation_count + 1}.png'), transformed["image"])
                cv2.imwrite(str(training_path / f'instrument{instrument_number}' / 'training' / 'ground_truth' / f'{file_name.stem}_{augmentation_count + 1}.png'), transformed["mask"])
