import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import torchvision.transforms as transforms
import os
import math
import shutil
from collections import defaultdict
from local_utils import get_dataset, get_extractor
from transformers import CLIPProcessor, CLIPModel

mean_dict = {
    'PACS' : [0.49400071, 0.41623791, 0.38352530]
}
std_dict = {
    'PACS' : [0.19193159, 0.16502413, 0.15799975]
}


def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))
    prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)
    
    for i in range(num_classes):
        prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot


def get_distance(features, labels):
    
    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
    
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels==i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)
    
    return distance


def get_features(args, model, processor, device):
    # obtain features of each sample
    # model = get_extractor(args)
    # model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    # model = model.to(args.device)
    
    global mean_dict
    global std_dict
    
    TRAIN_MEAN = mean_dict[args.dataset]
    TRAIN_STD = std_dict[args.dataset]
    
    
    if args.model == "CLIP":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            #transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
    elif "DINO" in args.model:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        
    data_train = get_dataset(args, transform, train=True)
    trainloader = DataLoader(data_train, batch_size=64, num_workers=5, pin_memory=True)
    
    targets, features = [], []
    for _, img, target in tqdm(trainloader):
        targets.extend(target.numpy().tolist())
        img = img.to(args.device)

        # Preprocess the images
        with torch.no_grad():
            if args.model == "CLIP":
                image_input = processor(images=img, return_tensors="pt", do_rescale=False).to(device)
                image_feature = model.get_image_features(**image_input).detach().cpu().numpy() # [1, 512]
                features.extend([image_feature[i] for i in range(image_feature.shape[0])])
            elif "DINO" in args.model:
                image_feature = model(img).cpu().numpy() # [1, 768]
                features.extend([image_feature[i] for i in range(image_feature.shape[0])])

        # Normalize the features
        # features = torch.cat(features, dim=0) # [N, 512]

        # feature = model(img).detach().cpu().numpy()
        # features.extend([feature[i] for i in range(feature.shape[0])])
        
    # features = torch.cat(features, dim=0).numpy() # [N, 512]
    features = np.array(features)
    targets = np.array(targets)
    
    return data_train, features, targets


def get_prune_idx(args, distance):
    
    low = 0.5 - args.rate / 2
    high = 0.5 + args.rate / 2
    
    sorted_idx = distance.argsort()
    low_idx = math.floor(distance.shape[0] * low)
    high_idx = math.ceil(distance.shape[0] * high)
    
    ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
    
    return ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--data_dir", type=str, default="data directory path")
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--model", type=str, default="CLIP")
    parser.add_argument("--save", default="index", help="dir to save pruned image ids")
    parser.add_argument("--rate", type=float, default=1, help="selection ratio")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    # CLIP
    if args.model == "CLIP":
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
        
    elif args.model == "DINO_base":
        processor = None
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)
        
    elif args.model == "DINO_small":
        processor = None
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device)

    model.eval()
    data_train, features, targets = get_features(args, model, processor, device)
    distance = get_distance(features, targets)
    cls_to_labels = defaultdict(list)
    cls_to_idxs = defaultdict(list)
    cls_to_distances = defaultdict(list)
    for idx, (path, label) in enumerate(data_train.imgs):
        # cls_to_labels[label].append(path)
        cls_to_idxs[label].append(idx)
        cls_to_distances[label].append(distance[idx])
    
    ids = []
    for klass in list(cls_to_distances.keys()):
        klass_ids = get_prune_idx(args, np.array(cls_to_distances[klass]))
        ids.extend(np.array(cls_to_idxs[klass])[klass_ids].tolist())
        
    print("len(ids)", len(ids))
    for index in ids:
        selected_file, label = data_train.imgs[index]
        target_file = selected_file.replace(args.dataset, f"{args.dataset}_{args.model}_moderate")
        os.makedirs("/".join(target_file.split("/")[:2]), exist_ok=True)
        os.makedirs("/".join(target_file.split("/")[:3]), exist_ok=True)
        os.makedirs("/".join(target_file.split("/")[:4]), exist_ok=True)
        shutil.copyfile(selected_file, target_file)

    os.makedirs(args.save, exist_ok=True)
    save = os.path.join(args.save, f"{args.dataset}.bin")
    with open(save, "wb") as file:
        pickle.dump(ids, file)

if __name__ == "__main__":
    main()