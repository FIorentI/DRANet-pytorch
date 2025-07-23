import os
import torch


def load_pretrained_model(pretrained_model_file, model, device, print_load_ok=True):
    print("Loading pretrained model (from provided location: " + pretrained_model_file + ")...")
    if os.path.isfile(pretrained_model_file):
        checkpoint = torch.load(pretrained_model_file, map_location=device)

        pretrained_dict = checkpoint
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict:
                if model_dict[k].shape == pretrained_dict[key].shape:
                    pretrained_keys.append(k)
                else:
                    skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if print_load_ok:
                print(k)
            model_dict[k] = pretrained_dict[key]

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file + " ...")

    model.to(device)


def load_pretrained_DANN_combined(encoder_file, classifier_file, model, device, print_load_ok=True):
    def load_weights(pretrained_model_file, submodel, submodel_name):
        print(f"Loading {submodel_name} weights from: {pretrained_model_file}")
        if os.path.isfile(pretrained_model_file):
            checkpoint = torch.load(pretrained_model_file, map_location=device)

            pretrained_dict = checkpoint
            model_dict = submodel.state_dict()

            pretrained_keys = []
            skipped_keys = []
            scratch_keys = []

            for k in model_dict.keys():
                if k in pretrained_dict:
                    if model_dict[k].shape == pretrained_dict[k].shape:
                        pretrained_keys.append(k)
                    else:
                        skipped_keys.append(k)
                else:
                    scratch_keys.append(k)

            if print_load_ok:
                print('-' * 80)
                print(f"{submodel_name} - Loaded keys:")
                for k in pretrained_keys:
                    print(k)

                print('-' * 80)
                print(f"{submodel_name} - Skipped (shape mismatch):")
                for k in skipped_keys:
                    print(f"{k} | model: {model_dict[k].shape}, pretrained: {pretrained_dict[k].shape}")

                print('-' * 80)
                print(f"{submodel_name} - Not found in pretrained:")
                for k in scratch_keys:
                    print(k)

            model_dict.update({k: pretrained_dict[k] for k in pretrained_keys})
            submodel.load_state_dict(model_dict)
        else:
            print(f"File not found: {pretrained_model_file}")

    # Charger les poids pour l'encodeur (c'est-à-dire toutes les couches avant la couche finale)
    load_weights(encoder_file, model, "Encoder")

    # Charger les poids pour le classificateur (c'est-à-dire la partie `fc` du modèle)
    load_weights(classifier_file, model, "Classifier")

    # S'assurer que le modèle est sur le bon appareil (GPU ou CPU)
    model.to(device)
    return model