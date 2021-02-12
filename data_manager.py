from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import pandas as pd
import random
from collections import Counter

from tqdm import tqdm

from video_loader import read_image
import transforms as T

"""Dataset classes"""


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root = './data/mars'

    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')
    attributes_path = osp.join(root, "mars_attributes.csv")
    #columns = ["action", "angle", "upcolor",
#     columns = ["upcolor",
#             "downcolor", "age", "up", "down", "bag",
#                           "backpack", "hat", "handbag", "hair",
#                           "gender", "btype"]
    columns = ["upcolor", "downcolor", "gender"]
    #attr_lens = [[5, 6], [9, 10, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    #attr_lens = [[],[9, 10, 4, 2, 2, 2, 2, 2, 2, 2, 2]]
    attr_lens = [[],[9, 10, 2]]   # just gender, upper.c, bottom.c
    
    def __init__(self, min_seq_len=0, attr=True, valid_ped_id_size=120, num_mmir_query_imgs=1000, num_mmir_query_videos=100):
        # validation_size = 2074, 625-470 = 155 persons, 2160 tracklets
        self._check_before_run()
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        self.attributes = pd.read_csv(self.attributes_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        
        # train and test have different pedestrains - 651 and 650; 1,261 different pedestrians captured by at least 2 cameras
        # so validation need to have different pedestrains too than train, to match test data
        # choose validation data from training set
        # validation data should be sampled with 'dense' as the test data
        # validation data is 25%, so train remains 6224
        np.random.seed(seed=24)
        print(len(set(track_train[:,2]))) # third column are the person-ids (625), 4th column is the cam-ids
        val_pids = np.random.choice(list(set(track_train[:,2])), valid_ped_id_size, replace=False) # since there are not all 1-625 person chosen, cant do random int, just select the distinct ids, and then random choice some from them
        val_IDX = np.where(track_train[:,2]==val_pids[0])
        # print(len(val_pids[1:]))
        for item in val_pids[1:]:
            val_IDX = np.append(val_IDX, np.where(track_train[:,2]==item))
        track_val = track_train[val_IDX,:]
        # print(len(track_val)) # 1454
        actual_train_IDX = [i for i in range(track_train.shape[0]) if i not in val_IDX]
        track_train = track_train[actual_train_IDX,:]
        print(np.shape(track_val))
        print(np.shape(track_train))
        val, num_val_tracklets, num_val_pids, num_val_imgs = \
          self._process_data(train_names, track_val, home_dir='bbox_train', relabel=False, min_seq_len=min_seq_len, attr=attr)
        
        
        # For each query, an average number of 3.7 cross-camera ground truths exist; each query has 4.2 image sequences that are captured under the same camera, and can be used as auxiliary information in addition to the query itself. 
        # 626 pedestrians
        # query and gallery are captured by different cameras, but not different person
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0

        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len, attr=attr)
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, attr=attr)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, attr=attr)
        
        np.random.seed(seed=24)
        mmir_img_query_IDX = np.random.choice(query_IDX, num_mmir_query_imgs, replace=False)
        mmir_video_query_IDX = np.random.choice(np.setdiff1d(query_IDX, mmir_img_query_IDX), num_mmir_query_videos, replace=False)
        track_mmir_img_query = track_test[mmir_img_query_IDX,:]
        track_mmir_video_query = track_test[mmir_video_query_IDX,:]
        mmir_img_query, num_mmir_img_query_tracklets, num_mmir_img_query_pids, num_mmir_img_query_imgs = \
          self._process_data(test_names, track_mmir_img_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, attr=attr)
        mmir_video_query, num_mmir_video_query_tracklets, num_mmir_video_query_pids, num_mmir_video_query_imgs = \
          self._process_data(test_names, track_mmir_video_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, attr=attr)
        self.mmir_img_query = mmir_img_query
        self.mmir_video_query = mmir_video_query
        self.num_mmir_img_query_pids = num_mmir_img_query_pids
        self.num_mmir_video_query_pids = num_mmir_video_query_pids

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  valid    | {:5d} | {:8d}".format(num_val_pids, num_val_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.valid = val

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_val_pids = num_val_pids
        
        print(num_mmir_img_query_pids, num_mmir_img_query_tracklets)
        print(num_mmir_video_query_pids, num_mmir_video_query_tracklets)

    def get_mean_and_var(self):
        imgs = []
        for t in self.train:
            imgs.extend(t[0])
        channel = 3
        x_tot = np.zeros(channel)
        x2_tot = np.zeros(channel)
        for img in tqdm(imgs):
            x = T.ToTensor()(read_image(img)).view(3, -1)
            x_tot += x.mean(dim=1).numpy()
            x2_tot += (x ** 2).mean(dim=1).numpy()

        channel_avr = x_tot / len(imgs)
        channel_std = np.sqrt(x2_tot / len(imgs) - channel_avr ** 2)
        print(channel_avr, channel_std)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
        if not osp.exists(self.attributes_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, attr=False):
        assert home_dir in ['bbox_train', 'bbox_test']
        # attributes = []
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            # if attr:
            #     if pid == 0 or pid == -1: continue
            # else:
            if pid == -1: continue # junk images are just ignored
            img_names = names[start_index-1:end_index]
            attribute = []
            if attr:
                t_id = int(img_names[0].split("F")[0].split("T")[1])
                # print(self.attributes)
                attribute = self.attributes[(self.attributes.person_id == pid) & (self.attributes.camera_id == camid) & (
                        self.attributes.tracklets_id == t_id)].values
                if len(attribute) > 0:
                    #attribute = attribute[0, 3:]
                    #attribute = attribute[0 : 2].tolist() + attribute[9:].tolist()
                    # They fucked it up here
                    # first line assigned some, then next line some other attributes, only 6 of them, thats why attrs.len is 6 later
                    ### uncommenting both line, taking all attr. except first 3, they are pid, camid, and track_id
                    ### because of pose and motion, attributes are 2D, but since i am ignoring those, have to take first elem
                    # attribute = attribute[0, 3:]
                    # attribute = attribute[0:2].tolist() + attribute[9:10].tolist() # just upper.c, bottom.c, gender; either one works, first 2 line or last line
                    attribute = attribute[0, 3:5].tolist() + attribute[0, -2:-1].tolist() # just upper.c, bottom.c, gender
                    
                  
                #print(attribute)
                #input()
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera*
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]

            if len(img_paths) >= min_seq_len:
                # random.shuffle(img_paths)
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, attribute)) ##### THis is the fuckling line that builds the dataset
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class DukeMTMC_Video(object):
    """
    DukeMTMC-vedio

    Reference:
    Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning. Wu et al., CVPR 2018

    Dataset statistics:
    702 identities (2,196 videos) for training and 702 identities (2,636 videos) for testing.
    # cameras: 8

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    # root = '/home/chenzy/datasets/mars'
    root = './data/duke'
    train_name_path = os.path.join(root, "train")
    gallery_name_path = os.path.join(root, "gallery")
    query_name_path = os.path.join(root, "query")
    attributes_path = osp.join(root, "duke_attributes.csv")

    columns = ["action", "angle", "backpack", "shoulder bag", "handbag", "boots", "gender", "hat", "shoes", "top", "downcolor", "topcolor"]
    # attr_lens = [[5, 6, 2, 2, 2, 2, 2, 2, 2],[2, 2, 2, 2, 2, 2, 2, 2, 8, 9]]
    attr_lens = [[5, 6], [2, 2, 2, 2, 2, 2, 2, 2, 8, 9]]
    
    def __init__(self, min_seq_len=0, attr=True):
        self._check_before_run()
        self.attributes = pd.read_csv(self.attributes_path)
        train, num_train_tracklets, num_train_pids, num_train_imgs, train_t_list = \
            self._process_data(self.train_name_path, relabel=True, min_seq_len=min_seq_len,
                               attr=attr)
        query, num_query_tracklets, num_query_pids, num_query_imgs, query_t_list = \
            self._process_data(self.query_name_path, relabel=False, min_seq_len=min_seq_len,
                               attr=attr)


        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, gallery_t_list = \
            self._process_data(self.gallery_name_path, relabel=False, min_seq_len=min_seq_len,
                               attr=attr, exclude_tracklets=query_t_list)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def get_mean_and_var(self):
        imgs = []
        for t in self.train:
            imgs.extend(t[0])
        channel = 3
        x_tot = np.zeros(channel)
        x2_tot = np.zeros(channel)
        for img in tqdm(imgs):
            # tmp = md.trn_ds.denorm(x).reshape(16, -1)
            # x = md.trn_ds.denorm(x).reshape(-1, 3)
            x = T.ToTensor()(read_image(img)).view(3, -1)
            x_tot += x.mean(dim=1).numpy()
            x2_tot += (x ** 2).mean(dim=1).numpy()

        channel_avr = x_tot / len(imgs)
        channel_std = np.sqrt(x2_tot / len(imgs) - channel_avr ** 2)
        print(channel_avr, channel_std)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.gallery_name_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_name_path))
        if not osp.exists(self.query_name_path):
            raise RuntimeError("'{}' is not available".format(self.query_name_path))
        # if not osp.exists(self.track_train_info_path):
        #     raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        # if not osp.exists(self.track_test_info_path):
        #     raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        # if not osp.exists(self.query_IDX_path):
        #     raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
        if not osp.exists(self.attributes_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, home_dir, relabel=False, min_seq_len=0, attr=False, exclude_tracklets=None):
        pid_list = []
        tracklets_path = []
        tracklets_list = []
        for p in os.listdir(home_dir):
            for t in os.listdir(os.path.join(home_dir, p)):
                if exclude_tracklets is None or t not in exclude_tracklets:
                    pid_list.append(int(p))
                    tracklets_path.append(os.path.join(home_dir, p + "/" + t))
                    tracklets_list.append(t)
        pid_list = set(pid_list)
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []
        for tracklet_idx in range(len(tracklets_path)):
            img_names = os.listdir(tracklets_path[tracklet_idx])
            pid = int(img_names[0].split("_")[0])
            camid = int(img_names[0].split("C")[1].split("_")[0])
            attribute = []
            if attr:
                t_id = int(tracklets_path[tracklet_idx].split("/")[-1])
                attribute = self.attributes[(self.attributes.person_id == pid) & (self.attributes.camera_id == camid) & (
                        self.attributes.tracklets_id == t_id)].values
                if len(attribute) > 0:
                    attribute = attribute[0, 3:]
                    attribute = attribute[0 : 2].tolist() + attribute[9:].tolist()
            assert 1 <= camid <= 8
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera*
            camnames = [img_name[6] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(tracklets_path[tracklet_idx], img_name) for img_name in img_names]

            if len(img_paths) >= min_seq_len:
                # random.shuffle(img_paths)
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, attribute))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, len(pid_list), num_imgs_per_tracklet, tracklets_list

"""Create dataset"""

__factory = {
    'mars': Mars,
    'duke':DukeMTMC_Video
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

if __name__ == '__main__':
    # test
    #dataset = Market1501()
    dataset = Mars()
    dataset.__init__()
    pass
