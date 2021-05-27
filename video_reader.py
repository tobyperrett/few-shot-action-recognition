import torch
from torch.hub import load
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor


class Split():
    """Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
    def __init__(self, folder, args):
        self.args = args
        
        self.gt_a_list = []
        self.videos = []


        class_folders = os.listdir(folder)
        class_folders = [f for f in class_folders if "." not in f]
        class_folders.sort()

        for class_folder in class_folders:
            video_folders = os.listdir(os.path.join(folder, class_folder))
            video_folders = [f for f in video_folders if "." not in f]
            video_folders.sort()
            for video_folder in video_folders:

                imgs = os.listdir(os.path.join(folder, class_folder, video_folder))
                imgs = [i for i in imgs if ((".jpg" in i) or (".png" in i))]
                if len(imgs) < self.args.seq_len:
                    continue            
                imgs.sort()
                paths = [os.path.join(folder, class_folder, video_folder, img) for img in imgs]
                paths.sort()
                class_id =  class_folders.index(class_folder)
                self.add_vid(paths, class_id)

        print("loaded {} videos from {}".format(len(self.gt_a_list), folder))

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_single_video(self, index):
        return self.videos[index], self.gt_a_list[index]

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def __len__(self):
        return len(self.gt_a_list)

class VideoDataset(torch.utils.data.Dataset):
    """Dataset for few-shot videos, which returns few-shot tasks. """
    def __init__(self, args, meta_batches=True):
        self.args = args
        self.get_item_counter = 0
        self.meta_batches = meta_batches

        self.split = "train"
        self.tensor_transform = transforms.ToTensor()

        self.train_split = Split(os.path.join(self.args.dataset, "train"), args)
        self.val_split = Split(os.path.join(self.args.dataset, "val"), args)
        self.test_split = Split(os.path.join(self.args.dataset, "test"), args)

        self.setup_transforms()


    def setup_transforms(self):
        """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
        video_transform_list = []
        video_test_list = []
            
        if self.args.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.args.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            raise NotImplementedError("img size transforms not setup")

        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.args.img_size))

        video_test_list.append(CenterCrop(self.args.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)
    

    def get_split(self):
        """ return the current split being used """
        if self.split == "train":
            return self.train_split
        elif self.split == "val":
            return self.val_split
        elif self.split == "test":
            return self.test_split

    def __len__(self):
        """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
        if self.meta_batches:
            return 1000000
        else:
            c = self.get_split()
            return len(c)
    
    def read_single_image(self, path):
        """Loads a single image from a specified path """
        with Image.open(path) as i:
            i.load()
            return i

    def load_and_transform_paths(self, paths):
        """ loads images from paths and applies transforms. Handles sampling if there are more frames than specified. """
        n_frames = len(paths)
        idx_f = np.linspace(0, n_frames-1, num=self.args.seq_len)
        idxs = [int(f) for f in idx_f]
        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.split == "train":
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs
    
    def get_seq(self, label, idx=-1):
        """Gets a single video sequence for a meta batch.  """
        c = self.get_split()
        if self.meta_batches:
            paths, vid_id = c.get_rand_vid(label, idx) 
            imgs = self.load_and_transform_paths(paths)
            
            return imgs, vid_id


    def get_meta_batch(self, index):
        """returns dict of support and target images and labels for a meta training task"""
        #select classes to use for this task
        c = self.get_split()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.args.way)

        if self.split == "train":
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
            for idx in idxs[self.args.shot:]:
                vid, vid_id = self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
        
        s = list(zip(support_set, support_labels))
        random.shuffle(s)
        support_set, support_labels = zip(*s)
        
        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)
        
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes}

    def get_single_vid(self, index):
        """gets a single video, used for pretraining the backbone"""
        c = self.get_split()
        paths, gt = c.get_single_video(index)
        vid = self.load_and_transform_paths(paths)

        #return {"images": vid, "target_labels": gt}
        return vid, gt

    def __getitem__(self, index):
        if self.meta_batches:
            return self.get_meta_batch(index)
        else:
            return self.get_single_vid(index)

if __name__ == "__main__":
    class Object(object):
        pass
    args = Object()
    args.dataset = "data/test"
    
    vd = VideoDataset(args)
