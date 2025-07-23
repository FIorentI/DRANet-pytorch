import os
from enum import Enum
import json


class ResizeInputPolicy(Enum):
    ResizeFix = 1
    IMAGE_NET = 2

    def __str__(self):
        return self.name


def get_dir_multiple_db_hdf5_no_validation(config_values):
    prefix_db_folder = "dataset_folder_"
    index_db = 1

    db_train_all = []
    db_test_all = []

    search_is_end = False

    while not search_is_end:
        name_db_folder = prefix_db_folder + str(index_db)

        if name_db_folder in config_values:
            dataset_folder = config_values[name_db_folder]

            db_train = os.path.join(dataset_folder, "train")
            directory_test = os.path.join(dataset_folder, "test")

            db_train_all.append(db_train)
            db_test_all.append(directory_test)

        else:
            search_is_end = True

        index_db += 1

    return db_train_all, db_test_all


def read_json_config_DANN(path_file):
    train_data_source = []
    train_data_target = []
    charset_path = ""
    dir_wandb = ""

    with open(path_file, "r") as fp:
        config_values = json.load(fp)
        print(config_values)

        # Extraction des chemins de données
        train_data_source = [[k, v] for k, v in config_values["train_data_source"].items()]
        train_data_target = [[k, v] for k, v in config_values["train_data_target"].items()]

        # Autres chemins
        charset_path = config_values.get("charset_file", "")
        dir_wandb = config_values.get("dir_wandb", "")

    return train_data_source, train_data_target, charset_path, dir_wandb

def read_json_config(path_file):
    train_info = []
    val_info = []
    test_info = []
    charset_path = ""
    dir_wandb = ""

    with open(path_file, "r") as fp:
        config_values = json.load(fp)

        print(config_values)

        if "train_data" in config_values:
            for key, value in config_values["train_data"].items():
                train_info.append([key, value])
        if "val_data" in config_values:
            for key, value in config_values["val_data"].items():
                val_info.append([key, value])
        if "test_data" in config_values:
            for key, value in config_values["test_data"].items():
                test_info.append([key, value])

        if "charset_file" in config_values:
            charset_path = config_values["charset_file"]

        # if "charset_files" in config_values:
        dir_wandb = config_values["dir_wandb"]

    return train_info, val_info, test_info, charset_path, dir_wandb


def read_json_config_eval_feature_space(path_file):
    data = []
    data_dagecc = []

    charset_path = ""
    dir_wandb = ""

    with open(path_file, "r") as fp:
        config_values = json.load(fp)

        print(config_values)

        if "data" in config_values:
            data = config_values["data"]

        if "data_dagecc" in config_values:
            data_dagecc = config_values["data_dagecc"]

        if "charset_file" in config_values:
            charset_path = config_values["charset_file"]

        # if "charset_files" in config_values:
        dir_wandb = config_values["dir_wandb"]

    return data, data_dagecc, charset_path, dir_wandb

