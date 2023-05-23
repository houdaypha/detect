import os
import yaml
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Config:
    """Read yaml config file"""

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'File {path} not found')
        with open(path) as file:
            yaml_config = yaml.load(file.read(), Loader=yaml.Loader)

        self.path = yaml_config['path']
        self.names = yaml_config['names']
        self.train = yaml_config.get('train', None)
        self.valid = yaml_config.get('val', None)
        self.test = yaml_config.get('test', None)


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, boxes, labels = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        # TODO: Remove
        totensor = transforms.ToTensor()
        image = totensor(image)
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)
        target = {
            "boxes": boxes,
            "labels": labels
        }
        return image, target


class CustomData:
    def __init__(self, path, batch=1, shuffle=False, workers=0):
        self.config: Config = Config(path)
        self.batch = batch
        self.shuffle = shuffle
        self.workers = workers

        if self.config.train:
            train_data = self.read_data(
                os.path.join(self.config.path, self.config.train))
            train_data = train_data[:16] # XXX: DEBUG
            self.train_datset = CustomDataset(train_data)
            self.train_loader = DataLoader(
                self.train_datset, batch_size=batch, shuffle=shuffle,
                num_workers=workers, collate_fn=collate_fn)

        if self.config.valid:
            valid_data = self.read_data(
                os.path.join(self.config.path,self.config.valid))
            valid_data = valid_data[:16] # XXX: DEBUG
            self.valid_datset = CustomDataset(valid_data)
            self.valid_loader = DataLoader(
                self.valid_datset, num_workers=workers, collate_fn=collate_fn)
        
        if self.config.test:
            test_data = self.read_data(
                os.path.join(self.config.path,self.config.test))
            test_data = test_data[:16] # XXX: DEBUG
            self.test_datset = CustomDataset(test_data)
            self.test_loader = DataLoader(self.test_datset, shuffle=False)

    def read_data(self, path):
        data = []
        for entry in tqdm(os.scandir(path)):
            # Reading image
            image = self.read_image(entry.path)
            w, h = image.size  # image.size -> (w, h)
            # XXX: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
            
            # # np: H x W x C, torch: C x H x W
            # image = np.array(image) # H x W x C
            # image = image.transpose((2, 0, 1))
            # image = torch.from_numpy(image)

            # Reading label
            # labels -> (boxes, classes)
            label_path = img2label_paths(entry.path)
            boxes, classes = self.read_label(label_path, w, h)
            data.append([image, boxes, classes])
        return data

    def read_image(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)  # PIL.Image.Image
            return img.convert("RGB")

    def read_label(self, path, imgw, imgh):
        with open(path, 'r') as f:
            lbs = [
                x.split()
                for x in f.read().strip().splitlines() if len(x)
            ]

        
        classes = np.array([lb[0] for lb in lbs], dtype=np.int64)
        classes = classes + 1 # label 0 for the background
        boxes = []
        for lb in lbs:
            x, y, w, h = map(lambda x: float(x), lb[1:])

            # xmin, ymin, xmax, ymax
            xmin = x - (w / 2)
            ymin = y - (h / 2)
            xmax = x + (w / 2)
            ymax = y + (h / 2)

            assert xmin >= 0, f'Negative value {xmin}, with label {path}'
            assert ymin >= 0, f'Negative value {ymin}, with label {path}'
            assert xmax >= 0, f'Negative value {xmax}, with label {path}'
            assert ymax >= 0, f'Negative value {ymax}, with label {path}'


            # Scale to image dimentions
            xmin = int(imgw * xmin)
            ymin = int(imgh * ymin)
            xmax = int(imgw * xmax)
            ymax = int(imgh * ymax)

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array(boxes, dtype=np.int64)
        return boxes, classes

# utils

def collate_fn(batch):
    return tuple(zip(*batch))


def img2label_paths(img_path):
    # /images/, /labels/ substrings
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'


# def read_label(path, imgw, imgh):
#     with open(path, 'r') as f:
#         lbs = [
#             x.split()
#             for x in f.read().strip().splitlines() if len(x)
#         ]

#     classes = np.array([lb[0] for lb in lbs], dtype=np.uint)
#     boxes = []
#     for lb in lbs:
#         x, y, h, w = map(lambda x: float(x), lb[1:])

#         # xmin, ymin, xmax, ymax
#         xmin = x - (w / 2)
#         ymin = y - (h / 2)
#         xmax = x + (w / 2)
#         ymax = y + (h / 2)

#         # Change scalle to image size
#         xmin = int(imgw * xmin)
#         ymin = int(imgh * ymin)
#         xmax = int(imgw * xmax)
#         ymax = int(imgh * ymax)

#         boxes.append([xmin, ymin, xmax, ymax])

#     boxes = np.array(boxes, dtype=np.uint)
#     return boxes, classes

if __name__ == '__main__':
    CustomData('./conf/torch.yaml', 4, False, 1)
