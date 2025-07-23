import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet34_Weights
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, gaussian_blur, rgb_to_grayscale, \
    InterpolationMode, resize, center_crop, convert_image_dtype
from torchvision.utils import save_image

from src.datautils.augmentation.erosion_dilatation_batch import Erosion2d, Dilation2d
from src.datautils.data_load import ResizeInputPolicy
from src.datautils.image.rescale_transform import rescale_fix_size_batch


class MultipleImageLabelDataset(Dataset):
    """
    Difference with v2:
        Tag are in char level
    """

    def __init__(self,
                 list_db: list,
                 fixed_size,
                 resize_config,
                 charset_dict_all,
                 ignore_index,
                 apply_augmentation,
                 config_augmentation,
                 grayscale_activate,
                 ratio_train_db=1.0,
                 pad_left=0,
                 pad_right=0):
        """
        """

        self.image_paths = []

        self.data = []
        self.labels_ind = []

        self.id_item = []

        self.apply_augmentation = apply_augmentation
        self.config_augmentation = config_augmentation

        self.fixed_size = fixed_size
        # self.transforms = transforms
        self.pad_left = pad_left
        self.pad_right = pad_right

        self.resize_config = resize_config

        dim_input_channel = 3

        if grayscale_activate == 1:
            dim_input_channel = 1

        # Initialize the Transforms for image NET -> to refactor clean
        # First test RestNet 34
        weights = ResNet34_Weights.DEFAULT
        self.preprocess_image_net = weights.transforms()

        self.erode_t = Erosion2d(dim_input_channel, dim_input_channel, 3, soft_max=True)
        self.dilate_t = Dilation2d(dim_input_channel, dim_input_channel, 3, soft_max=True)

        self.threshold_apply_data_augmentation = 0.0

        for one_db_path in list_db:
            dbfile = open(one_db_path[1], 'rb')
            db = pickle.load(dbfile)

            data_one_db = db["data"]  # [:number_to_keep].clone()
            label_one_db = db["label"]  # [:number_to_keep].clone()
            class_to_idx = db["class_to_idx"]

            list_index_item = []
            list_index_target = []

            # # Get list index source
            for key, index_v in class_to_idx.items():
                index_item = label_one_db == index_v
                list_index_item.append(index_item)

                if key not in charset_dict_all:
                    # EMNIST upper and lower are the same, label in lower case
                    key_upper = key.upper()

                    if key_upper in charset_dict_all:
                        # label_one_db[label_one_db == index_v] = charset_dict_all[key_upper]
                        # index_item = label_one_db == index_v
                        # list_index_item.append(index_item)

                        list_index_target.append(charset_dict_all[key_upper])
                    else:
                        # label_one_db[label_one_db == index_v] = ignore_index
                        list_index_target.append(ignore_index)
                else:
                    # label_one_db[label_one_db == index_v] = charset_dict_all[key]
                    list_index_target.append( charset_dict_all[key])

            # Update target index
            for index_item, i_t in zip(list_index_item, list_index_target):
                label_one_db[index_item] = i_t

            print(one_db_path)
            print("Size origin: " + str(data_one_db.shape))
            print()

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
            # save_image(data_one_db, 'C:/Users/simcor/dev/logs/img_digit_img_net.png')

            # Pad height, width
            if self.resize_config == ResizeInputPolicy.ResizeFix:
                data_one_db = rescale_fix_size_batch(data_one_db, self.fixed_size[0], self.fixed_size[1], pad_value=0)
                # save_image(data_one_db, 'C:/Users/simcor/dev/logs/img_digit_resize.png')
            # # KO CPU RAM BERZELIUS
            # elif self.resize_config == ResizeInputPolicy.IMAGE_NET:
            #     data_one_db = preprocess_image_net(data_one_db)
            #     # save_image(data_one_db, 'C:/Users/simcor/dev/logs/img_digit_img_net.png')
            else:
                data_one_db = F.pad(input=data_one_db,
                                    pad=(0, fixed_size[0] - shape_db[2], 0, fixed_size[1] - shape_db[3], 0, 0, 0, 0),
                                    mode='constant', value=0)

            if ratio_train_db != 1.0:
                nb_item_filter = int(ratio_train_db * shape_db[0])

                all_index = torch.randperm(shape_db[0])

                index_filter = all_index[:nb_item_filter]

                data_one_db = data_one_db[index_filter]
                label_one_db = label_one_db[index_filter]

            self.data.append(data_one_db)
            self.labels_ind.append(label_one_db)

        self.data_origin = self.data
        self.labels_ind_origin = self.labels_ind

        self.data = torch.cat(self.data)
        self.labels_ind = torch.cat(self.labels_ind)

        self.ratio_data_origin = 1.0

    def set_threshold_apply_data_augmentation(self, value):
        self.threshold_apply_data_augmentation = value
        print("threshold_apply_data_augmentation: " + str(self.threshold_apply_data_augmentation))

    def reduce_data_train_label(self, new_ratio):
        self.ratio_data_origin = new_ratio

        data_origin_filter = []
        label_origin_filter = []

        for one_db, one_label in zip(self.data_origin, self.labels_ind_origin):
            nb_item = one_db.shape[0]

            nb_item_filter = int(self.ratio_data_origin * nb_item)

            print("Nb DB origin:" + str(nb_item))
            print("Nb DB filter:" + str(nb_item_filter))

            all_index = torch.randperm(nb_item)

            index_filter = all_index[:nb_item_filter]

            data_one_db = one_db[index_filter]
            label_one_db = one_label[index_filter]

            data_origin_filter.append(data_one_db)
            label_origin_filter.append(label_one_db)

        self.data = torch.cat(data_origin_filter)
        self.labels_ind = torch.cat(label_origin_filter)

    def add_unlabel_data(self, new_data, new_label):
        new_data = new_data.to(self.data.device)
        new_label = new_label.to(self.labels_ind.device)

        data_origin_filter = []
        label_origin_filter = []

        for one_db, one_label in zip(self.data_origin, self.labels_ind_origin):
            nb_item = one_db.shape[0]
            nb_item_filter = int(self.ratio_data_origin * nb_item)

            all_index = torch.randperm(nb_item)

            index_filter = all_index[:nb_item_filter]

            data_one_db = one_db[index_filter]
            label_one_db = one_label[index_filter]

            data_origin_filter.append(data_one_db)
            label_origin_filter.append(label_one_db)

        data_origin_filter = torch.cat(data_origin_filter)
        label_origin_filter = torch.cat(label_origin_filter)

        self.data = torch.cat((data_origin_filter, new_data))
        self.labels_ind = torch.cat((label_origin_filter, new_label))

        print("New nb item:" + str(self.data.shape[0]))

    def __len__(self):
        """
        Returns the number of images in the dataset
        Returns
        -------
        length: int
            number of images in the dataset
        """

        # return len(self.image_paths)
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        """

        img = self.data[idx]  # tensor

        if self.resize_config == ResizeInputPolicy.IMAGE_NET:
            # img *= 255.0
            # save_image(img, 'C:/Users/simcor/dev/logs/img_digit.png')
            # img = self.preprocess_image_net(img)

            img = resize(img, 256, interpolation=InterpolationMode.BILINEAR, antialias=True)
            img = center_crop(img, 224)
            # if not isinstance(img, Tensor):
            #     img = F.pil_to_tensor(img)
            img = convert_image_dtype(img, torch.float)
            # save_image(img, 'C:/Users/simcor/dev/logs/img_digit_img_net.png')

        # save_image(img, 'C:/Users/simcor/dev/logs/character_classification/img_digit_before.png')
        if self.apply_augmentation:

            if np.random.rand() > self.threshold_apply_data_augmentation:
                # https://pytorch.org/vision/0.15/transforms.html
                # apply random combination of transformation
                aug = self.config_augmentation  # params["config"]["augmentation"]

                # Apply transform random composition
                # Erosion
                if "erosion" in aug.keys() and np.random.rand() < aug["erosion"]["proba"]:
                    # save_image(img, 'C:/Users/simcor/dev/logs/img_digit_before.png')
                    img_erode = img.unsqueeze(0)  # add batch dim

                    img_erode = self.erode_t(img_erode)

                    img_erode = img_erode.squeeze(0)  # remove batch dim
                    img = img_erode.detach()  # remove grad link to erode transform

                    # save_image(img, 'C:/Users/simcor/dev/logs/img_digit_after.png')
                # Dilatation
                if "dilatation" in aug.keys() and np.random.rand() < aug["dilatation"]["proba"]:
                    # save_image(img, 'C:/Users/simcor/dev/logs/img_digit_before.png')
                    img_dilate = img.unsqueeze(0)  # add batch dim

                    img_dilate = self.dilate_t(img_dilate)

                    img_dilate = img_dilate.squeeze(0)  # remove batch dim
                    img = img_dilate.detach()  # remove grad link to erode transform

                    # save_image(img, 'C:/Users/simcor/dev/logs/img_digit_after.png')
                # Contrast
                if "contrast" in aug.keys() and np.random.rand() < aug["contrast"]["proba"]:
                    factor = np.random.uniform(aug["contrast"]["min_factor"], aug["contrast"]["max_factor"])
                    img = adjust_contrast(img, factor)
                # Bright
                if "brightness" in aug.keys() and np.random.rand() < aug["brightness"]["proba"]:
                    factor = np.random.uniform(aug["brightness"]["min_factor"], aug["brightness"]["max_factor"])
                    img = adjust_brightness(img, factor)
                # Gaussian Blur
                if "gaussian_blur" in aug.keys() and np.random.rand() < aug["gaussian_blur"]["proba"]:
                    img = gaussian_blur(img, kernel_size=(3, 7), sigma=(0.3, 4.))

                if "sign_flipping" in aug.keys() and np.random.rand() < aug["sign_flipping"]["proba"]:
                    img = 1 + (-1 * img)

                # img = np.array(img)
                # img = torch.as_tensor(img, dtype=torch.float32)
        # save_image(img, 'C:/Users/simcor/dev/logs/character_classification/img_digit_after.png')

        return img, self.labels_ind[idx]
