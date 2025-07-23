import glob
import os
import pickle
from random import random, shuffle

from skimage import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torchvision.models import ResNet34_Weights
from torchvision.transforms.functional import rgb_to_grayscale

from torchvision.utils import save_image

from src.datautils.data_load import ResizeInputPolicy
from src.datautils.image.rescale_transform import rescale_fix_size_batch


class ImageDataset(Dataset):
    """
    Difference with v2:
        Tag are in char level
    """

    def __init__(self,
                 db_path: list,
                 fixed_size,
                 resize_config,
                 grayscale_activate,
                 transforms: list = None,
                 tag: str = "NotSet"):
        """
        """

        self.image_paths = []

        self.data = []
        self.labels_ind = []

        self.id_item = []

        self.fixed_size = fixed_size
        self.transforms = transforms

        self.resize_config = resize_config

        # Initialize the Transforms for image NET -> to refactor clean
        # First test RestNet 34
        weights = ResNet34_Weights.DEFAULT
        self.preprocess_image_net = weights.transforms()

        dbfile = open(db_path, 'rb')
        db = pickle.load(dbfile)

        data_one_db = db["data"]

        shape_db = data_one_db.shape

        max_value = torch.max(data_one_db)
        # Some db are already normalize
        if max_value > 1:
            data_one_db = data_one_db / 255.0

        # N, Channel, Height, Width
        if len(shape_db) == 4:
            if grayscale_activate == 1:
                if shape_db[1] == 3:
                    data_one_db = rgb_to_grayscale(data_one_db)
            else:
                if shape_db[1] == 1:
                    data_one_db = data_one_db.repeat(1, 3, 1, 1)

        # N, Height, Width
        elif len(shape_db) == 3:
            data_one_db = data_one_db.unsqueeze(1)

            if grayscale_activate == 0:
                data_one_db = data_one_db.repeat(1, 3, 1, 1)

        shape_db = data_one_db.shape

        if self.resize_config == ResizeInputPolicy.ResizeFix:
            data_one_db = rescale_fix_size_batch(data_one_db, self.fixed_size[0], self.fixed_size[1], pad_value=0)
        # elif self.resize_config == ResizeInputPolicy.IMAGE_NET:
        #     data_one_db = preprocess_image_net(data_one_db)
        else:
            data_one_db = F.pad(input=data_one_db,
                                pad=(0, fixed_size[0] - shape_db[2], 0, fixed_size[1] - shape_db[3], 0, 0, 0, 0),
                                mode='constant', value=0)

        self.data = data_one_db

        # path_save_img = "C:/Users/simcor/dev/data/Digits/imgs_" + tag + ".png"
        # save_image(self.data, path_save_img)

    def __len__(self):
        """
        Returns the number of images in the dataset
        Returns
        -------
        length: int
            number of images in the dataset
        """

        #return len(self.image_paths)
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        """
        img = self.data[idx]  # tensor

        if self.resize_config == ResizeInputPolicy.IMAGE_NET:
            # img *= 255.0
            # save_image(img, 'C:/Users/simcor/dev/logs/img_digit.png')
            img = self.preprocess_image_net(img)

        return img  #, self.labels_ind[idx]
