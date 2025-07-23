import os
import pickle

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

import numpy as np


def extract_and_save_svhn(dir_save):
    training_data = datasets.SVHN(
        root="data",
        download=True,
        transform=ToTensor(),
        split="train"
    )

    index_label = np.unique(training_data.labels)
    # Create class_to_idx
    class_to_idx = {}

    for index_l in index_label:
        label_str = str(index_l)
        class_to_idx[label_str] = index_l

    path_train = os.path.join(dir_save, "train")

    db = {
        "data": torch.from_numpy(training_data.data),
        "label": torch.from_numpy(training_data.labels),
        "class_to_idx": class_to_idx
    }

    dbfile = open(path_train, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()

    test_data = datasets.SVHN(
        root="data",
        download=True,
        transform=ToTensor(),
        split="test"
    )

    path_test = os.path.join(dir_save, "test")

    db = {
        "data": torch.from_numpy(test_data.data),
        "label": torch.from_numpy(test_data.labels),
        "class_to_idx": class_to_idx
    }

    # Its important to use binary mode
    dbfile = open(path_test, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()


def extract_and_save_emnist(dir_save, split_version):
    dir_save_split = os.path.join(dir_save, split_version)
    os.makedirs(dir_save_split, exist_ok=True)

    training_data = datasets.EMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        split=split_version
    )

    path_train = os.path.join(dir_save_split, "train")

    data = training_data.data
    data = torch.permute(data, (0, 2, 1))

    db = {
        "data": data,
        "label": training_data.targets,
        "class_to_idx": training_data.class_to_idx
    }

    dbfile = open(path_train, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()

    test_data = datasets.EMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        split=split_version
    )

    path_test = os.path.join(dir_save_split, "test")

    data = test_data.data
    data = torch.permute(data, (0, 2, 1))

    db = {
        "data": data,
        "label": test_data.targets,
        "class_to_idx": test_data.class_to_idx
    }

    dbfile = open(path_test, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()


# Create subset to debug on CPU
def create_debug_dataset(path_origin, path_new, number_to_keep):
    dbfile = open(path_origin, 'rb')
    db = pickle.load(dbfile)

    data_d = db["data"].clone()
    label_d = db["label"].clone()

    all_index = torch.randperm(data_d.shape[0])
    index_filter = all_index[:number_to_keep]

    data_d = data_d[index_filter]
    label_d = label_d[index_filter]

    dbfile.close()

    db_debug = {
        "data": data_d,
        "label": label_d,
        "class_to_idx": db["class_to_idx"]
    }

    dbfile_debug = open(path_new, 'ab')

    # source, destination
    pickle.dump(db_debug, dbfile_debug)
    dbfile_debug.close()


def save_tensor_as_img(path_data, dir_save):

    dbfile = open(path_data, 'rb')
    db = pickle.load(dbfile)

    index = 0
    for one_img in db["data"]:
        one_img_n = one_img / 255
        path_save = os.path.join(dir_save, str(index) + ".png")
        save_image(one_img_n, path_save)

        index += 1


if __name__ == '__main__':

    working_dir = "C:/Users/simcor/dev/data/Digits/"

    dir_save_svhn = os.path.join(working_dir, "SVHN")
    os.makedirs(dir_save_svhn, exist_ok=True)
    extract_and_save_svhn(dir_save_svhn)

    svhn_train = os.path.join(dir_save_svhn, "train")
    svhn_train_debug = os.path.join(dir_save_svhn, "train_debug")

    create_debug_dataset(svhn_train, svhn_train_debug, 33)

    svhn_test = os.path.join(dir_save_svhn, "test")
    svhn_test_debug = os.path.join(dir_save_svhn, "test_debug")

    create_debug_dataset(svhn_test, svhn_test_debug, 33)


    # # EMNIST
    # dir_save_emnist = os.path.join(working_dir, "EMNIST")
    # os.makedirs(dir_save_emnist, exist_ok=True)
    #
    # dir_save_emnist_letter = os.path.join(working_dir, "EMNIST", "letters")
    # dir_save_emnist_img = os.path.join(dir_save_emnist_letter, "img")
    # os.makedirs(dir_save_emnist_img, exist_ok=True)
    # path_data_emnist = os.path.join(dir_save_emnist_letter, "test")

    # Check visually the data
    # save_tensor_as_img(path_data_emnist, dir_save_emnist_img)

    # # extract_and_save_emnist(dir_save_emnist, "digits")
    # emnist = "C:/Users/simcor/dev/data/Digits/EMNIST/digits/train"
    # emnist_debug = "C:/Users/simcor/dev/data/Digits/EMNIST/digits/train_debug"
    #
    # create_debug_dataset(emnist, emnist_debug, 33)
    #
    # emnist = "C:/Users/simcor/dev/data/Digits/EMNIST/digits/test"
    # emnist_debug = "C:/Users/simcor/dev/data/Digits/EMNIST/digits/test_debug"
    #
    # create_debug_dataset(emnist, emnist_debug, 33)

    # Letter
    # extract_and_save_emnist(dir_save_emnist, "letters")
    # emnist = "C:/Users/simcor/dev/data/Digits/EMNIST/letters/train"
    # emnist_debug = "C:/Users/simcor/dev/data/Digits/EMNIST/letters/train_debug"
    #
    # create_debug_dataset(emnist, emnist_debug, 33)
    #
    # emnist = "C:/Users/simcor/dev/data/Digits/EMNIST/letters/test"
    # emnist_debug = "C:/Users/simcor/dev/data/Digits/EMNIST/letters/test_debug"
    #
    # create_debug_dataset(emnist, emnist_debug, 33)

