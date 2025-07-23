import argparse
import faulthandler
import json
import os
import time

import torch
import wandb
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.datautils.collate.classif_batch_collate import CollateImageLabelClassification
from src.datautils.data_load import ResizeInputPolicy, read_json_config
from src.datautils.datasets.multiple_datasets_classif import MultipleImageLabelDataset
from src.datautils.datasets.one_dataset_classification import OneImageLabelDataset
from src.datautils.text.charset_token import CharsetToken
from src.evaluate.evaluate_cnn_one_epoch import evaluate_cnn_one_epoch
from src.models.cnn.resnet_torchvision import CNNModel, ResNetTorch, BasicBlock, Bottleneck
from src.models.model_utils import load_pretrained_model

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

config_values = {}

with open(args.config_file, "r") as fp:
    config_values = json.load(fp)

# Paths
directory_log = args.log_dir
_, val_info, test_info, charset_path, dir_wandb = read_json_config(args.config_file)

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

# Default validation -> first db
val_db = OneImageLabelDataset(val_info[0][1],
                              fixed_size_img,
                              args.resize_config,
                              char_dict,
                              ignore_index,
                              grayscale_activate=args.grayscale_activate)

print('Nb samples val {}:'.format(len(val_db)))

val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                            collate_fn=collate_fn, shuffle=False)

all_test_dataloader = []
dbs_test = []
for db in test_info:
    test_db = OneImageLabelDataset(db[1],
                                   fixed_size_img,
                                   args.resize_config,
                                   char_dict,
                                   ignore_index,
                                   grayscale_activate=args.grayscale_activate)

    print('Nb samples test {}:'.format(len(test_db)))

    test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                 collate_fn=collate_fn, shuffle=False)

    all_test_dataloader.append(test_dataloader)
    dbs_test.append(db[0])

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

print_summary = ""

loss_classif = nn.CrossEntropyLoss(ignore_index=ignore_index)

# Validation
f1_macro_total, f1_per_class, loss_val_total = evaluate_cnn_one_epoch(model, val_dataloader, device, loss_classif, f1_score)

print('F1-Macro : {:.2f} %'.format(100 * f1_macro_total))
print('Val Loss : ' + str(loss_val_total))
# print("F1 per class:")
# print(char_list)        # to update with the correct charset
# print(f1_per_class)

nb_char = len(char_list)
nb_class_val = len(f1_per_class)

for i_c in range(nb_char):
    if i_c < nb_class_val:
        print(char_list[i_c] + f": {f1_per_class[i_c]:.2f}")


print("--------Testing-----------")
for i_db in range(len(test_info)):
    print("--------------------------------------")
    print(dbs_test[i_db])

    f1_macro_total, f1_per_class, loss_val_total = evaluate_cnn_one_epoch(model,
                                                                          all_test_dataloader[i_db],
                                                                          device,
                                                                          loss_classif,
                                                                          f1_score,
                                                                          print_first_batch=True,
                                                                          char_list=char_list)

    print('F1-Macro : {:.2f} %'.format(100 * f1_macro_total))
    # print('Accuracy : {} %'.format(100 * correct / total))
    print('Val Loss : ' + str(loss_val_total))
    # print("F1 per class:")
    # print(char_list)        # to update with the correct charset
    # print(f1_per_class)

    nb_char = len(char_list)
    nb_class_val = len(f1_per_class)

    for i_c in range(nb_char):
        if i_c < nb_class_val:
            print(char_list[i_c] + f": {f1_per_class[i_c]:.2f}")

print("End")

