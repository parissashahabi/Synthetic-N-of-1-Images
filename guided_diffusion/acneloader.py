import torch
import torch.nn
import numpy as np
import os
import os.path
from torch.utils.data import Dataset
from monai import transforms

class AcneDataset(Dataset):
    def __init__(self, data_dir, transform=None, severity_levels=None, test_flag=False, img_size=256):
        """
        Using your existing transform structure from the notebook
        """
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.test_flag = test_flag
        self.img_size = img_size
        
        # Use your transforms if none provided
        if transform is None:
            if test_flag:
                self.transform = self._get_val_transforms()
            else:
                self.transform = self._get_train_transforms()
        else:
            self.transform = transform
        
        # Collect image files and labels (same as before)
        self.image_files = []
        self.labels = []
        
        severity_levels = severity_levels or [0, 1, 2, 3]
        severity_folders = [f"acne{level}_1024" for level in severity_levels]
        
        for folder in severity_folders:
            folder_path = os.path.join(data_dir, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist.")
                continue
                
            severity = int(folder.split('_')[0].replace('acne', ''))
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                self.image_files.append(file_path)
                self.labels.append(severity)
        
        print(f"Loaded {len(self.image_files)} images")
        # Print class distribution
        from collections import Counter
        print(f"Class distribution: {dict(Counter(self.labels))}")

    def _get_train_transforms(self):
        """Your existing train transforms from the notebook"""
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=0, 
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.Resized(
                keys=["image"],
                spatial_size=(self.img_size, self.img_size),
                mode="bilinear",
            ),
            # Apply 90-degree rotation to every image
            transforms.Rotated(
                keys=["image"],
                angle=np.pi/2,
                keep_size=True,
            ),
            # Apply horizontal flip to every image
            transforms.Flipd(
                keys=["image"],
                spatial_axis=1,
            ),
            transforms.EnsureTyped(keys=["image"]),
        ])

    def _get_val_transforms(self):
        """Your existing val transforms from the notebook"""
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=0, 
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.Resized(
                keys=["image"],
                spatial_size=(self.img_size, self.img_size),
                mode="bilinear",
            ),
            # Apply 90-degree rotation to every image  
            transforms.Rotated(
                keys=["image"],
                angle=np.pi/2,
                keep_size=True,
            ),
            # Apply horizontal flip to every image
            transforms.Flipd(
                keys=["image"],
                spatial_axis=1,
            ),
            transforms.EnsureTyped(keys=["image"]),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        filename = os.path.basename(img_path)
        
        # Create sample dictionary for MONAI transforms
        sample = {"image": img_path, "label": label}
        
        # Apply your transforms
        if self.transform:
            sample = self.transform(sample)
            
        # Extract the transformed image
        image_tensor = sample["image"]
        
        # Create output dictionary compatible with original code
        out_dict = {"y": label}
        
        # Return in the expected format: (image, out_dict, weak_label, dummy_segmentation, filename)
        dummy_segmentation = torch.zeros(self.img_size, self.img_size)
        
        return (image_tensor, out_dict, label, dummy_segmentation, filename)