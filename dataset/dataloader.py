from PIL import Image
import torch.utils.data as data
import os


class SeqDataset(data.Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        self.data_dir: str = data_dir
        self.transform = transform
        self.data: list = []
        self.idx: list = []
        # save all the name of images in data_dir to self.data
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                self.data.append(os.path.join(root, file))
                idx = file.split("_")[-1].split(".")[0]
                # convert string to int
                self.idx.append(int(idx))

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.idx[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class TestDataset(data.Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        self.data_dir: str = data_dir
        self.transform = transform
        self.data: list = []
        self.idx: list = []
        # save all the name of images in data_dir to self.data
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                self.data.append(os.path.join(root, file))
                idx = file.split("_")[-1].split(".")[0]
                # convert string to int
                self.idx.append(int(idx))

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.idx[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_path

    def __len__(self):
        return len(self.data)
