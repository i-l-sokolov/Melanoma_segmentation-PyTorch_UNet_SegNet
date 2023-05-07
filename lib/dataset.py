import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torch
import os


def get_cropflip():
    crop_flip = transforms.Compose([
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.75, 1), antialias=False),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    return crop_flip


def get_augmentation():
    augmentation = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
        transforms.RandomGrayscale(),
        transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])
    return augmentation


def get_valid_tranforms():
    valid_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return valid_transforms


class SegDataset(Dataset):
    def __init__(self, folders, mode):
        self.folders = folders
        self.mode = mode

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mask_resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)

        self.augmentation = get_augmentation()
        self.crop_flip = get_cropflip()
        self.valid_transforms = get_valid_tranforms()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        image_name = os.path.basename(folder) + '.bmp'
        image_path = os.path.join(folder, os.path.basename(folder) + '_Dermoscopic_Image',
                                  os.path.basename(folder) + '.bmp')

        mask_path = os.path.join(folder, os.path.basename(folder) + '_lesion',
                                 os.path.basename(folder) + '_lesion.bmp')

        if self.mode == 'train':

            image = self.to_tensor(self.augmentation(Image.open(image_path)))
            mask = self.to_tensor(Image.open(mask_path))
            concat = torch.concatenate([image, mask])
            concat = self.crop_flip(concat)
            image, mask = concat[:3], concat[3:]
            image = self.normalize(image)
            mask = (mask > 0.5).to(torch.int8)

        else:
            image = self.valid_transforms(Image.open(image_path))
            mask = self.to_tensor(self.mask_resize(Image.open(mask_path)))
            mask = (mask > 0.5).to(torch.int8)
        return image, mask
