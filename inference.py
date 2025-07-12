import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from utils import *
from gaussian_splatting_func import *
from adapter import *
from projector_func import *
from data import *
from clip_preprocess import *
from inference_func import *
from config import *


current_dir = os.path.dirname(__file__)
checkpoint = os.path.join(current_dir , 'best_model_0.9558.pth')
h5_dir = os.path.join(current_dir , 'modelnet40_ply_hdf5_2048')
save_dir = os.path.join(current_dir , 'modelnet40_npy')


#####################
# save modelnet40_ply_hdf5_2048 to modelnet40_npy if npy does not exist
if config['exist_flag']:
    h5_dir = h5_dir
    save_dir = save_dir
    shape_names_path = os.path.join(h5_dir, "shape_names.txt")
    convert_modelnet40_h5_to_npy(h5_dir, save_dir, shape_names_path,max_samples=config['max_samples'])
#####################


seed = 42
seed_func(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/16", device=device)


clip_preprocess = ProcessorGradientFlow()


root_dir = os.path.join(current_dir , 'modelnet40_npy')


classnames = sorted(os.listdir(root_dir))
dataset = ModelNet40PointCloudDataset(
    root_dir=root_dir,
    classnames=classnames,
    split='test'
)

# test loader 생성
loader = DataLoader(
    dataset,
    batch_size=1,           # 필요에 따라 조절
    shuffle=False,
    num_workers=0,           # 시스템에 맞게 조정
    drop_last=False
)

hidden_dim = 512
distance_cutoff = 0.9
importance_score = (0.5,1.0)
relative_normalized_weight = 0.1

predicted_classes = inference_from_checkpoint(
    checkpoint_path=checkpoint,
    loader=loader,
    processor_gradient_flow=clip_preprocess,
    clip_model=clip_model,
    classnames=classnames,
    device=device,
    hidden_dim = 512,
    distance_cutoff = 0.9,
    importance_score = (0.5,1.0),
    relative_normalized_weight = 0.1
)

visualize(loader , predicted_classes , classnames)