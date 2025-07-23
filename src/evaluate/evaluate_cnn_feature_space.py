import argparse
import faulthandler
import os
import time
import math
from math import floor

import numpy as np
import torch
import umap
from torch.utils.data import DataLoader
import cv2
from sklearn.manifold._t_sne import TSNE

from src.datautils.collate.classif_batch_collate import CollateImageLabelClassification
from src.datautils.data_load import ResizeInputPolicy, read_json_config_eval_feature_space
from src.datautils.datasets.one_dataset_classification import OneImageLabelDataset
from src.datautils.text.charset_token import CharsetToken
from src.models.cnn.resnet_torchvision import CNNModel, ResNetTorch, BasicBlock, Bottleneck
from src.models.model_utils import load_pretrained_model
from src.visualisation.print_features_space import get_features_reduction

parser = argparse.ArgumentParser()

parser.add_argument("config_file")
parser.add_argument("log_dir")

parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--debug_pc', default=0, type=int)
parser.add_argument("--path_model", default="", help="path of pretrained model", type=str)
parser.add_argument('--height_max', default=48, type=int)
parser.add_argument('--width_max', default=48, type=int)

parser.add_argument('--grayscale_activate', default=1, type=int)

parser.add_argument('--cnn_model', type=lambda tw: CNNModel[tw], choices=list(CNNModel), default=CNNModel.ResNet34)

parser.add_argument('--resize_config', type=lambda tw: ResizeInputPolicy[tw], choices=list(ResizeInputPolicy),
                    default=ResizeInputPolicy.ResizeFix)
print("===============================================================================")

begin = time.time()
args = parser.parse_args()
print(args)

faulthandler.enable()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device :")
print(device)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))

# Paths
directory_log = args.log_dir
data, data_dagecc, charset_path, dir_wandb = read_json_config_eval_feature_space(args.config_file)

# Alphabet
charset = CharsetToken(charset_path)
nb_char_all = charset.get_nb_char()

# For class not present in charset
ignore_index = nb_char_all + 1
charset.add_label("IGNORE")

char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()
nb_char_all = charset.get_nb_char()

# Data
fixed_size_img = (args.height_max, args.width_max)

# Pad img with black = 0
c_collate_fn = CollateImageLabelClassification(imgs_pad_value=[0])
collate_fn = c_collate_fn.collate_fn

data_db = OneImageLabelDataset(data,
                               fixed_size_img,
                               args.resize_config,
                               char_dict,
                               ignore_index,
                               grayscale_activate=args.grayscale_activate)

print('Nb samples {}:'.format(len(data_db)))

data_dataloader = DataLoader(data_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                             collate_fn=collate_fn, shuffle=False)

data_dagecc_db = OneImageLabelDataset(data_dagecc,
                                      fixed_size_img,
                                      args.resize_config,
                                      char_dict,
                                      ignore_index,
                                      grayscale_activate=args.grayscale_activate)

print('Nb samples dagecc {}:'.format(len(data_dagecc_db)))

data_dagecc_dataloader = DataLoader(data_dagecc_db, num_workers=args.num_workers, batch_size=args.batch_size,
                                    pin_memory=True,
                                    collate_fn=collate_fn, shuffle=False)

# Model
nb_channel_input = 3
if args.grayscale_activate == 1:
    nb_channel_input = 1

if args.cnn_model == CNNModel.ResNet18:
    model = ResNetTorch(BasicBlock, [2, 2, 2, 2], nb_channel_input=nb_channel_input, num_classes=nb_char_all)
if args.cnn_model == CNNModel.ResNet34:
    model = ResNetTorch(BasicBlock, [3, 4, 6, 3], nb_channel_input=nb_channel_input, num_classes=nb_char_all)
if args.cnn_model == CNNModel.ResNet50:
    model = ResNetTorch(Bottleneck, [3, 4, 6, 3], nb_channel_input=nb_channel_input, num_classes=nb_char_all)

if os.path.isfile(args.path_model):
    load_pretrained_model(args.path_model, model, device)

print(f"Transferring model to {str(device)}...")
model = model.to(device)
number_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {number_parameters:,} trainable parameters.")

n_neighbors_umap = 15
min_dist_umap = 0.1

# reduction = umap.UMAP(n_neighbors=n_neighbors_umap, n_components=2, min_dist=min_dist_umap, random_state=42)

perplexity = 10.0
reduction = TSNE(n_components=2,
                 perplexity=perplexity,
                 n_jobs=1)
# features_data = get_features_reduction(model, data_dataloader, device, reduction_umap)
# features_data_dageec = get_features_reduction(model, data_dagecc_dataloader, device, reduction_umap)

features_data = get_features_reduction(model, data_dataloader, data_dagecc_dataloader, device, reduction)

height_img = 4000
width_img = 4000

margin = 50

image = np.zeros(shape=(height_img + 2 * margin, width_img + 2 * margin, 3), dtype=np.int16)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
# BGR
color_data = (255, 0, 0)
color_data_dagecc = (0, 255, 0)

# for label_index, coordinates in features_data.items():
#     label = char_list[label_index]
#     for one_coordinate_point_norm in coordinates:
#         one_coordinate_point = (floor(one_coordinate_point_norm[1] * height_img) + margin,
#                                 floor(one_coordinate_point_norm[0] * width_img) + margin)
#         image = cv2.putText(image, label, one_coordinate_point, font, fontScale, color_data, 2, cv2.LINE_AA)

for label_index, coordinates in features_data.items():
    label = char_list[label_index]
    # for one_coordinate_point_norm in coordinates:
    for one_coordinate_point_norm_index_data in coordinates:
        one_coordinate_point_norm, index_data = one_coordinate_point_norm_index_data
        if not np.isnan(one_coordinate_point_norm).all():

            current_color = color_data

            if index_data == 1:
                current_color = color_data_dagecc

            one_coordinate_point = (floor(one_coordinate_point_norm[1] * height_img) + margin,
                                    floor(one_coordinate_point_norm[0] * width_img) + margin)
            image = cv2.putText(image, label, one_coordinate_point, font, fontScale, current_color, 2, cv2.LINE_AA)
        else:
            print("NaN coordinates")

# cv2.imshow("test", image)
filename = os.path.join(directory_log, "features_2d.png")
cv2.imwrite(filename, image)
print("End")
