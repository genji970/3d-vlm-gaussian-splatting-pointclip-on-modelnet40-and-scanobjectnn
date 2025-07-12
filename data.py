import os
import numpy as np
import torch
import clip
from torch.utils.data import Dataset, DataLoader

class ModelNet40PointCloudDataset(Dataset):
    def __init__(self, root_dir, classnames, split='train'):
        """
        root_dir: '/content/drive/MyDrive/modelnet40_npy'
        classnames: ['airplane', 'chair', ...]
        split: 'train' or 'test'
        """
        self.samples = []
        self.classnames = classnames
        self.split = split  # 'train' or 'test'

        for class_id, cls_name in enumerate(classnames):
            split_dir = os.path.join(root_dir, cls_name, split)
            if not os.path.exists(split_dir):
                continue
            npy_files = sorted(os.listdir(split_dir))
            for f in npy_files:
                full_path = os.path.join(split_dir, f)
                self.samples.append((full_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        pc = np.load(npy_path)  # (N, 3)
        return torch.tensor(pc, dtype=torch.float32), label

from collections import defaultdict
from torch.utils.data import Subset
import numpy as np

def split_few_shot_dataset(dataset, num_per_class=16, seed=42, debug=False):
    from collections import Counter, defaultdict
    import numpy as np
    from torch.utils.data import Subset

    np.random.seed(seed)

    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label, indices in class_to_indices.items():
        np.random.shuffle(indices)
        train_indices.extend(indices[:num_per_class])
        val_indices.extend(indices[num_per_class:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset