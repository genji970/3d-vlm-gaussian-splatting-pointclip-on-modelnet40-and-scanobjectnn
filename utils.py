from torchvision.transforms.functional import gaussian_blur
import torch

def fill_depth_map(depth_map: torch.Tensor, sigma=2.0, steps=3):
    x = depth_map.unsqueeze(1)  # (B, 1, H, W)
    for _ in range(steps):
        # torchvision expects (B, C, H, W)
        blurred = gaussian_blur(x, kernel_size=[11, 11], sigma=[sigma, sigma])
        x = torch.max(x, blurred)
    return x.squeeze(1)


def naive_fps(points, n_samples):
    B, N, _ = points.shape
    device = points.device
    #print(f"[naive_fps] points: {points.shape}, n_samples: {n_samples}")  # 🔍 shape 디버그

    # 샘플 초기화
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), float('inf'), device=device)
    #print(f"[naive_fps] init selected: {selected.shape}, dist: {dist.shape}")  # 🔍 shape 디버그

    farthest = torch.randint(0, N, (B,), device=device)  # 처음 점 랜덤 선택
    batch_indices = torch.arange(B, device=device)
    #print(f"[naive_fps] farthest: {farthest.shape}, batch_indices: {batch_indices.shape}")  # 🔍 shape 디버그

    for i in range(n_samples):
        selected[:, i] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)  # (B, 1, 3)
        dists = torch.norm(points - centroid, dim=-1)  # (B, N)
        dist = torch.min(dist, dists)  # 최소 거리 업데이트
        farthest = torch.max(dist, dim=1)[1]

        #if i % 10 == 0 or i == n_samples - 1:
        #    print(f"[naive_fps] step {i}: selected: {selected[:, :i+1].shape}, centroid: {centroid.shape}, dist: {dist.shape}")  # 🔍 shape 디버그

    return selected  # (B, n_samples)

import numpy as np
import torch
from collections import Counter
import matplotlib.pyplot as plt

def visualize(test_loader, predicted_classes, classnames):
    true_classes = []
    for _, labels in test_loader:
        true_classes.extend(
            labels.tolist() if isinstance(labels, torch.Tensor) else [labels]
        )

    # 텐서일 경우 리스트로 변환
    if isinstance(predicted_classes, torch.Tensor):
        predicted_classes = predicted_classes.tolist()

    # 예시 출력
    for i, c in enumerate(predicted_classes):
        print(f"[{i}] → 예측 클래스: {classnames[c]}")

    # flatten이 필요할 경우만 수행 (예: 2D list)
    if isinstance(predicted_classes[0], (list, np.ndarray, torch.Tensor)):
        predicted_classes = [p for batch in predicted_classes for p in batch]

    pred_counter = Counter(predicted_classes)
    true_counter = Counter(true_classes)

    all_classes = list(range(len(classnames)))
    pred_counts = [pred_counter.get(i, 0) for i in all_classes]
    true_counts = [true_counter.get(i, 0) for i in all_classes]

    x = np.arange(len(all_classes))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, true_counts, width, label='True Labels')
    plt.bar(x + width / 2, pred_counts, width, label='Predicted Labels')
    plt.xticks(x, classnames, rotation=45, ha='right')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution of True vs Predicted Classes")
    plt.legend()
    plt.tight_layout()
    plt.show()


import random
import os

def seed_func(seed):
    seed = seed
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU 연산
    torch.cuda.manual_seed(seed)  # GPU 연산
    torch.cuda.manual_seed_all(seed)  # Multi-GPU 연산
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(42)

import os
import h5py
import numpy as np
from tqdm import tqdm

def load_classnames(txt_path):
    with open(txt_path, 'r') as f:
        classnames = [line.strip() for line in f.readlines()]
    return classnames

def convert_modelnet40_h5_to_npy(h5_dir, save_dir, shape_names_path, max_samples=100):
    import h5py
    os.makedirs(save_dir, exist_ok=True)
    classnames = load_classnames(shape_names_path)
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]

    count = 0  # 변환된 총 샘플 수

    for fname in tqdm(h5_files, desc="Converting H5 to NPY"):
        if count >= max_samples:
            break

        fpath = os.path.join(h5_dir, fname)
        is_train = 'train' in fname.lower()
        split = 'train' if is_train else 'test'

        with h5py.File(fpath, 'r') as f:
            data = f['data'][:]   # (B, 2048, 3)
            label = f['label'][:] # (B, 1)

            for i in range(data.shape[0]):
                if count >= max_samples:
                    break

                pc = data[i]
                lbl = int(label[i, 0])
                cls_name = classnames[lbl]

                # 저장 경로 설정
                cls_split_dir = os.path.join(save_dir, cls_name, split)
                os.makedirs(cls_split_dir, exist_ok=True)

                filename = f"{fname.replace('.h5','')}_{i}.npy"
                np.save(os.path.join(cls_split_dir, filename), pc)
                count += 1

    print(f"✅ Saved {count} samples.")





