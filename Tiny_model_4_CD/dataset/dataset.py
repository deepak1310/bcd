from typing import List, Tuple
from collections.abc import Sized
from os.path import join
import albumentations as alb
from torchvision.transforms import Normalize

import numpy as np
import torch
from matplotlib.image import imread
from torch.utils.data import Dataset
from torch import Tensor


class MyDataset(Dataset, Sized):
    def __init__(
        self,
        data_path: str,
        mode: str,
    ) -> None:
        """
        data_path: Folder containing the sub-folders:
            "A" for test images,
            "B" for ref images, 
            "label" for the gt masks,
            "list" containing the image list files ("train.txt", "test.txt", "eval.txt").
        """
        # Store the path data path + mode (train,val,test):
        self._mode = mode
        self._A = join(data_path, "A")
        self._B = join(data_path, "B")
        self._label = join(data_path, "label3")

        # In all the dirs, the files share the same names:
        self._list_images = self._read_images_list(data_path)

        # Initialize augmentations:
        if mode == 'train':
            self._augmentation = _create_shared_augmentation()
            self._aberration = _create_aberration_augmentation()
        
        # Initialize normalization:
        self._normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def __getitem__(self, indx):
        # Current image set name:
        imgname = self._list_images[indx].strip('\n')

        # Loading the images:
        x_ref = imread(join(self._A, imgname))
        x_test = imread(join(self._B, imgname))
        x_mask = _binarize(imread(join(self._label, imgname)))

        # Data augmentation in case of training:
        if self._mode == "train":
            #print("WWWWWWWWWWWWWWWWWWWWW")
            x_ref, x_test, x_mask = self._augment(x_ref, x_test, x_mask)

        # Trasform data from HWC to CWH:
        x_ref, x_test, x_mask = self._to_tensors(x_ref, x_test, x_mask)

        return (x_ref, x_test), x_mask

    def __len__(self):
        return len(self._list_images)

    def _read_images_list(self, data_path: str) -> List[str]:
        images_list_file = join(data_path,'list', self._mode + ".txt")
        with open(images_list_file, "r") as f:
            return f.readlines()
    
    def _augment(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # First apply augmentations in equal manner to test/ref/x_mask:
        transformed = self._augmentation(image=x_ref, image0=x_test, x_mask0=x_mask)
        x_ref = transformed["image"]
        x_test = transformed["image0"]
        x_mask = transformed["x_mask0"]

        # Then apply augmentation to single test ref in different way:
        x_ref = self._aberration(image=x_ref)["image"]
        x_test = self._aberration(image=x_test)["image"]

        return x_ref, x_test, x_mask
    
    def _to_tensors(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self._normalize(torch.tensor(x_ref).permute(2, 0, 1)),
            self._normalize(torch.tensor(x_test).permute(2, 0, 1)),
            torch.tensor(x_mask),
        )

def _ncreate_shared_augmentation():
    aug1 = alb.HorizontalFlip(p=0.5)
    aug2 = alb.VerticalFlip(p=0.5)
    aug3 = alb.RandomRotate90(p=0.5)
    aug4 = alb.Transpose(p=0.5)
    aug5 = alb.Rotate(limit=5, p=0.5)
    aug5 = alb.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    aug6 = alb.GridDistortion(p=0.5)
    aug7 = alb.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
    return alb.Compose([alb.OneOf([aug1,aug2,aug3,aug4,aug5,aug6,aug7],p=0.7)],additional_targets={"image0": "image", "x_mask0": "mask"},)

def _ncreate_aberration_augmentation():
    aug1 = alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,p=0.5)
    #print("-------->")
    aug2 = alb.GaussianBlur(blur_limit=[3, 5], p=0.5)
    #aug3 = alb.Equalize(p=0.5)
    aug4 = alb.Sharpen(alpha=(0.1, 0.3),p=0.5) 
    aug5 = alb.UnsharpMask(p=0.5)
    aug6 = alb.GaussNoise(var_limit=(3, 7), p=0.5)
    return alb.Compose([alb.SomeOf([aug1,aug2,aug4,aug5,aug6],2,p=0.7)])
    

def _create_shared_augmentation():
    return alb.Compose(
        [
            alb.Flip(p=0.5),
            alb.Rotate(limit=5, p=0.5),
        ],
        additional_targets={"image0": "image", "x_mask0": "mask"},
    )

def _create_aberration_augmentation():
    return alb.Compose([
        alb.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        alb.GaussianBlur(blur_limit=[3, 5], p=0.5),
    ])

def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask * 255, 0, 1).astype(int)
