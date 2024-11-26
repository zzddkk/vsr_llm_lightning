import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
from torch.utils.data import Dataset


class video_dataset(Dataset):
    def __init__(self, root_dir,video_transform,labels="label",subset="train",data_ratio=0.1):
        self.root_dir = root_dir
        self.labels = os.path.join(root_dir,labels)
        self.subset = subset
        self.data_ratio = data_ratio
        self.list = self.load_list()
        if subset == "train":
            self.list = self.list[:int(len(self.list)*data_ratio)]
        self.list.reverse()
        self.video_transform = video_transform

    def __len__(self):
        return len(self.list)
    
    def load_data(self,index):
        path = self.list[index][1]
        with open(os.path.join(self.root_dir,"lrs2",path.replace(".mp4",".txt")).replace("video","text")) as f:
            target = f.read().splitlines()[0]
        video = torchvision.io.read_video(os.path.join(self.root_dir,"lrs2",path),pts_unit="sec",output_format="THWC")[0]
        video = video.permute((0, 3, 1, 2)) # T,H,W,C -> T,C,H,W
        video = self.video_transform(video)
        return {"video":video.float(),"target":target,"file_path":path}
        
    def load_list(self):
        data_list = [] 
        labels_path = os.listdir(self.labels)
        for label in labels_path:
            if self.subset in label:
                with open(os.path.join(self.labels,label)) as f:
                    data_list.extend(map(lambda x:(x.split(",")[0],x.split(",")[1],int(x.split(",")[2]),torch.tensor([int(_) for _ in x.split(",")[3].split()])),f.read().splitlines()))
                data_list.sort(key=lambda x:x[2],reverse=False)
        # print(f"Found {len(data_list)} videos in {self.subset} subset")
        # if self.subset == "train":
            # print(f"Using {int(len(data_list)*self.data_ratio)} videos in {self.subset} subset to train")
        return data_list

    def __getitem__(self, idx):
        data = self.load_data(idx)
        return data