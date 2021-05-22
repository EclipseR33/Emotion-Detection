import torch
from torch.utils.data import Dataset

from PIL import Image
from glob import glob
import os


class MMAFacialExpressionDataSet(Dataset):
    def __init__(self, root, type='train', transform=None, need_pil_image=False):
        self.root = os.path.join(root, type)
        self.transform = transform
        self.need_pil_image = need_pil_image

        self.class_root = glob(self.root + '\\*')
        self.classes = [c.split('\\')[-1] for c in self.class_root]

        self.image_set = {'image': [], 'label': []}

        for c in self.class_root:
            images = glob(c + '\\*')

            c = c.split('\\')[-1]
            l = self.classes.index(c)
            label = [l for _ in range(len(images))]

            self.image_set['image'] += images
            self.image_set['label'] += label

        self.len = len(self.image_set['image'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 全部转为灰度图处理
        image_path, label = self.image_set['image'][idx], self.image_set['label'][idx]

        pil_image = Image.open(image_path)
        image = pil_image.convert('L')

        if self.transform is not None:
            image = self.transform(image)
        image /= 255.
        label = torch.tensor(label)

        if self.need_pil_image:
            return image, label, pil_image
        else:
            return image, label


class PredictDataSet(Dataset):
    def __init__(self, root, transform=None, need_pil_image=False):
        self.root = root
        self.transform = transform
        self.need_pil_image = need_pil_image

        self.image_set = {'image': []}

        images = glob(self.root + '\\*')

        self.image_set['image'] += images

        self.len = len(self.image_set['image'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 全部转为灰度图处理
        image_path = self.image_set['image'][idx]

        pil_image = Image.open(image_path)
        image = pil_image.convert('L')

        if self.transform is not None:
            image = self.transform(image)
        image /= 255.

        if self.need_pil_image:
            return image, pil_image
        else:
            return image


if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(48),
        transforms.ToTensor()
    ])
    root = r'F:\AI\KaggleData\MMA FACIAL EXPRESSION'
    dataset = MMAFacialExpressionDataSet(root, type='train', transform=transform)

    # root = '../inference/images'
    # dataset = PredictDataSet(root, transform)

    dataset[0]
    print(dataset.classes)
