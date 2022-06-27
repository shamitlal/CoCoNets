# import tensorflow as tf
import numpy as np
import scipy
import torch
import pickle
from torch.utils.data import DataLoader
import hyperparams as hyp
import utils.py
import utils.misc
import random
import utils.geom
random.seed(0)
from backend import readers
import os, json, random, imageio
import utils.improc
import ipdb
import time 
st = ipdb.set_trace
np.set_printoptions(precision=2, suppress=True)

class IndexedDataset(torch.utils.data.Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # img, label = self.base[idx]
        # return (img, label, idx)
        feed = self.base[idx]
        return (feed, idx)

class NpzDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, shuffle, data_format, data_consec, seqlen):
        print('dataset_path = %s' % dataset_path)
        with open(dataset_path) as f:
            content = f.readlines()
        dataset_location = dataset_path.split('/')[:-1]
        dataset_location = '/'.join(dataset_location)
        print('dataset_loc = %s' % dataset_location)
        
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
 
        self.records = records
        self.shuffle = shuffle
        self.data_format = data_format
        self.data_consec = data_consec
        self.seqlen = seqlen

    def __getitem__(self, index):

        if hyp.idx0_dataloader:
            index = 0

        if (self.data_format=='seq' or
            self.data_format=='multiview' or
            self.data_format=='nuscenes' or
            self.data_format=='cdtb' or
            self.data_format=='ktrack' or
            self.data_format=='kodo' or
            self.data_format=='traj' or
            self.data_format=='intphys_single' or
            self.data_format=='zebrafish_multiview' or
            self.data_format=='intphys_fullseq' or
            self.data_format=='surveil_traj' or
            self.data_format=='cater_traj' or
            self.data_format=='cater_traj_w_shape' or
            self.data_format=='surveil_multiview' or
            self.data_format=='complete' or
            self.data_format=='oldtraj' or
            self.data_format=='simpletraj'
            or self.data_format=='3dq'):
            filename = self.records[index]
            d = np.load(filename, allow_pickle=True)
            d = dict(d)
            # elif self.data_format=='nuscenes':
            #     filename = self.records[index]
            #     d = np.load(filename, allow_pickle=True)['data']
            #     d = d[()]
        else:
            assert(False) # reader not ready yet

        # print("data format: ", self.data_format)
        # if the sequence length > 2, select S frames
        if self.shuffle:
            d = self.random_select_single(d, num_samples=self.seqlen)
        else:
            d = self.non_random_select_single(d, num_samples=self.seqlen)

        if hyp.do_time_flip:
            d = self.random_time_flip_single(d)

        if self.data_format=='simpletraj':
            # for k, v in d.items():
            #     print(k)
            rgb_camX0 = d['rgb_camX0']
            # move channel dim inward, like pytorch wants
            rgb_camX0 = np.transpose(rgb_camX0, axes=[2, 0, 1])
            rgb_camX0 = utils.py.preprocess_color(rgb_camX0)
            d['rgb_camX0'] = rgb_camX0

        elif self.data_format=='cdtb':
            rgb_camXs = d['rgb_camXs']
            rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
            rgb_camXs = utils.py.preprocess_color(rgb_camXs)
            d['rgb_camXs'] = rgb_camXs

        elif self.data_format=='nuscenes':
            # for k, v in d.items():
            #     print(k)
            #     print(v.shape)
            #     # print(k, v.shape)
            # print('rgb_camXs', rgb_camXs.shape)
            
            # move channel dim inward, like pytorch wants
            rgb_camXs = d['rgb_camXs']
            rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
            rgb_camXs = utils.py.preprocess_color(rgb_camXs)
            d['rgb_camXs'] = rgb_camXs
            pass
        elif self.data_format=='3dq':
            
            rgb_camXs = d['rgb_camXs_raw']
            rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
            rgb_camXs = utils.py.preprocess_color(rgb_camXs)
            d['rgb_camXs'] = rgb_camXs

            d['pix_T_cams'] = d['pix_T_cams_raw']
            d['xyz_camXs'] = d['xyz_camXs_raw']
            d['origin_T_camXs'] = d['origin_T_camXs_raw']
            d['origin_T_camRs'] = utils.geom.safe_inverse(torch.tensor(d['camR_T_origin_raw'])).numpy()
            if hyp.do_tsdf_implicit_occ:
                pickle_fname = filename.split('/')[-1]
                tsdf_filename = os.path.join(hyp.tsdf_dataset_dir, f"tsdf_{pickle_fname}")
                tsdf_data = pickle.load(open(tsdf_filename, "rb"))
                
                d['tsdf_inside'] = utils.misc.get_random_elements(tsdf_data['inside'], 4*250000)
                d['tsdf_outside'] = utils.misc.get_random_elements(tsdf_data['outside'], 2*250000)
                d['tsdf_sdf1'] = utils.misc.get_random_elements(tsdf_data['sdf1'], 2*250000)

        elif self.data_format=='complete':
            rgb_cam0s = d['rgb_cam0s']
            # move channel dim inward, like pytorch wants
            rgb_cam0s = np.transpose(rgb_cam0s, axes=[0, 3, 1, 2])
            rgb_cam0s = utils.py.preprocess_color(rgb_cam0s)
            d['rgb_cam0s'] = rgb_cam0s

            rgb_cam1s = d['rgb_cam1s']
            # move channel dim inward, like pytorch wants
            rgb_cam1s = np.transpose(rgb_cam1s, axes=[0, 3, 1, 2])
            rgb_cam1s = utils.py.preprocess_color(rgb_cam1s)
            d['rgb_cam1s'] = rgb_cam1s
            
            rgb_cam2s = d['rgb_cam2s']
            # move channel dim inward, like pytorch wants
            rgb_cam2s = np.transpose(rgb_cam2s, axes=[0, 3, 1, 2])
            rgb_cam2s = utils.py.preprocess_color(rgb_cam2s)
            d['rgb_cam2s'] = rgb_cam2s
            
            rgb_cam3s = d['rgb_cam3s']
            # move channel dim inward, like pytorch wants
            rgb_cam3s = np.transpose(rgb_cam3s, axes=[0, 3, 1, 2])
            rgb_cam3s = utils.py.preprocess_color(rgb_cam3s)
            d['rgb_cam3s'] = rgb_cam3s
            
            rgb_camRs = d['rgb_camRs']
            # move channel dim inward, like pytorch wants
            rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
            rgb_camRs = utils.py.preprocess_color(rgb_camRs)
            d['rgb_camRs'] = rgb_camRs
            
        else:
            rgb_camXs = d['rgb_camXs']
            # move channel dim inward, like pytorch wants
            rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
            rgb_camXs = utils.py.preprocess_color(rgb_camXs)
            d['rgb_camXs'] = rgb_camXs
        
        if (self.data_format=='multiview'):
            # we also have camR
            rgb_camRs = d['rgb_camRs']
            rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
            rgb_camRs = utils.py.preprocess_color(rgb_camRs)
            d['rgb_camRs'] = rgb_camRs
        # for key,val in d.items():
        #     try:
        #         print(key,val.shape)
        #     except Exception:
        #         print(key,val)
        

        if hyp.pseudo_multiview:
            list_of_keys = ['0','1','2','3']
            sampled = random.sample(list_of_keys,2)
            pix_T_cams = np.concatenate((d['pix_T_cams'],d['pix_T_cams']),axis=0)
            rgb_camXs = np.concatenate((d['rgb_cam%ss'%sampled[0]],d['rgb_cam%ss'%sampled[1]]),axis=0)
            xyz_camXs = np.concatenate((d['xyz_cam%ss'%sampled[0]],d['xyz_cam%ss'%sampled[1]]),axis=0)
            origin_T_camXs = np.concatenate((d['origin_T_cam%ss'%sampled[0]],d['origin_T_cam%ss'%sampled[1]]),axis=0)
            origin_T_camRs = np.concatenate((d['origin_T_cam%ss'%'R'],d['origin_T_cam%ss'%'R']),axis=0)
            rgb_camRs = d['rgb_camRs']
            # st()

            full_boxlist_camR = d['full_boxlist_camR']
            full_scorelist_s = d['full_scorelist']
            full_tidlist_s = d['full_tidlist']            

            ind_along_S = sampled
            if hyp.make_dense:
                all_xyz_camXs = []
                all_rgb_camXs = []
                all_origin_T_camXs = []
                # st()
                for key in list_of_keys:
                    all_xyz_camXs.append(d['xyz_cam%ss'%key])
                    all_rgb_camXs.append(d['rgb_cam%ss'%key])
                    all_origin_T_camXs.append(d['origin_T_cam%ss'%key])
                all_xyz_camXs = np.concatenate(all_xyz_camXs,axis=0)
                all_rgb_camXs = np.concatenate(all_rgb_camXs,axis=0)
                all_origin_T_camXs = np.concatenate(all_origin_T_camXs,axis=0)
            # st()
            # st()
            # st()
            d = {}
            d['pix_T_cams'] = pix_T_cams
            d['rgb_camXs'] = rgb_camXs
            d['xyz_camXs'] = xyz_camXs
            d['origin_T_camXs'] = origin_T_camXs
            d['origin_T_camRs'] = origin_T_camRs
            try:
                d['all_rgb_camXs'] = all_rgb_camXs
            except Exception as e:
                pass
            if hyp.make_dense:
                d['all_origin_T_camXs'] = all_origin_T_camXs
                d['all_xyz_camXs'] = all_xyz_camXs

            d['full_boxlist_camR'] = full_boxlist_camR
            d['full_scorelist'] = full_scorelist_s
            d['full_tidlist'] = full_tidlist_s

            d['ind_along_S'] = ind_along_S
            # st()
            # pix_T_cams (2, 4, 4)
            # origin_T_camRs (2, 4, 4)
            # origin_T_camXs (2, 4, 4)
            # rgb_camXs (2, 3, 256, 768)
            # xyz_camXs (2, 49152, 3)
            # lrtlist_camRs (2, 50, 19)
            # scorelist_s (2, 50)
            # tidlist_s (2, 50)
            # ind_along_S (2,)0.5570118


        if hyp.pseudo_traj:
            list_of_keys = ['0','1','2','3']
            sampled = random.sample(list_of_keys,1)
            sampled =  ['1']
            pix_T_cams = d['pix_T_cams']
            origin_T_camRs = d['origin_T_camRs']
            rgb_camRs = d['rgb_camRs']
            origin_T_camXs = d['origin_T_cam%ss'%sampled[0]]
            rgb_camXs = d['rgb_cam%ss'%sampled[0]]
            xyz_camXs = d['xyz_cam%ss'%sampled[0]]

            score_traj = d['score_traj']
            
            if hyp.use_lrt_not_box:
                lrt_traj_camR = d['lrt_traj_camR']
            else:
                box_traj_camR = d['box_traj_camR']
                full_boxlist_camR = d['full_boxlist_camR']

            full_scorelist_s = d['full_scorelist']
            full_tidlist_s = d['full_tidlist']

            ind_along_S = d['ind_along_S']
    
            if hyp.make_dense:
                all_xyz_camXs = []
                all_rgb_camXs = []
                all_origin_T_camXs = []
                for key in list_of_keys:
                    all_xyz_camXs.append(d['xyz_cam%ss'%key])
                    all_rgb_camXs.append(d['rgb_cam%ss'%key])
                    all_origin_T_camXs.append(d['origin_T_cam%ss'%key])
                all_xyz_camXs = np.stack(all_xyz_camXs,axis=1)
                all_rgb_camXs = np.stack(all_rgb_camXs,axis=1)
                all_origin_T_camXs = np.stack(all_origin_T_camXs,axis=1)            
            # st()
            
            d = {}
            d['pix_T_cams'] = pix_T_cams
            d['rgb_camRs'] = rgb_camRs
            d['rgb_camXs'] = rgb_camXs
            d['xyz_camXs'] = xyz_camXs
            d['origin_T_camXs'] = origin_T_camXs
            d['origin_T_camRs'] = origin_T_camRs
            d['score_traj'] = score_traj
            d['full_scorelist_s'] = full_scorelist_s
            d['full_tidlist_s'] = full_tidlist_s
            d['ind_along_S'] = ind_along_S
            if hyp.use_lrt_not_box:
                d['lrt_traj_camR'] = lrt_traj_camR
            else:
                d['box_traj_camR'] = box_traj_camR
                d['full_boxlist_camR'] = full_boxlist_camR
            if hyp.make_dense:
                d['all_rgb_camXs'] = all_rgb_camXs
                d['all_origin_T_camXs'] = all_origin_T_camXs
                d['all_xyz_camXs'] = all_xyz_camXs            

            # pix_T_cams (50, 4, 4)
            # origin_T_camRs (50, 4, 4)
            # origin_T_camXs (50, 4, 4)
            # rgb_camXs (50, 3, 256, 768)
            # xyz_camXs (50, 49152, 3)
            # lrt_traj_camR (50, 19)
            # score_traj (50,)
            # full_lrtlist_camRs (50, 50, 19)
            # full_scorelist_s (50, 50)
            # full_tidlist_s (50, 50)
            # ind_along_S range(0, 50)
        d['filename'] = filename
        return d

    def __len__(self):
        return len(self.records)

    def get_item_names(self):
        if self.data_format=='seq':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'boxlists',
                'tidlists',
                'scorelists',
            ]
        elif self.data_format=='multiview':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'seg_camXs',
                'rgb_camRs',
                'xyz_camXs',
                'boxlists',
                'tidlists',
                'scorelists',
            ]
        elif self.data_format=='traj':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'box_traj_camR',
                'score_traj',
                'full_boxlist_camR',
                'full_scorelist',
                'full_tidlist',
            ]
        elif self.data_format=='intphys_single':
            item_names = [
                'pix_T_camXs',
                'origin_T_camXs',
                'origin_T_camRs',
                'rgb_camXs',
                'xyz_camXs',
                'boxlist_camXs',
                'scorelist_s',
                'vislist_s',
                'tidlist_s',
            ]
        elif self.data_format=='zebrafish_multiview':
            item_names = [
                'pix_T_camXs',
                'world_T_camXs',
                'rgb_camXs',
                'centroids',
            ]
        elif self.data_format=='intphys_fullseq':
            item_names = [
                'pix_T_camXs',
                'origin_T_camXs',
                'origin_T_camRs',
                'rgb_camXs',
                'xyz_camXs',
                'boxlist_camXs',
                'scorelist_s',
                'vislist_s',
                'tidlist_s',
            ]
        elif self.data_format=='surveil_traj':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'lrt_traj_camR',
                'score_traj',
                'full_lrtlist_camRs',
                'full_scorelist_s',
                'full_tidlist_s',
            ]
        elif self.data_format=='surveil_multiview':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'lrtlist_camRs',
                'scorelist_s',
                'tidlist_s',
            ]
        elif self.data_format=='cater_traj':
            item_names = [
                'pix_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'world_T_camXs',
                'lrt_traj_world',
                'scorelist',
                'world_T_camR',
            ]
        elif self.data_format=='cater_traj_w_shape':
            item_names = [
                'pix_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'world_T_camXs',
                'lrt_traj_world',
                'scorelist',
                'world_T_camR',
                'shapelist',
                'actionlist',
            ]
        elif self.data_format=='complete':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_cam0s',
                'origin_T_cam1s',
                'origin_T_cam2s',
                'origin_T_cam3s',
                'rgb_camRs',
                'rgb_cam0s',
                'rgb_cam1s',
                'rgb_cam2s',
                'rgb_cam3s',
                'xyz_camRs',
                'xyz_cam0s',
                'xyz_cam1s',
                'xyz_cam2s',
                'xyz_cam3s',
                'score_traj',
                'full_scorelist',
                'full_tidlist',
            ]
            if hyp.use_lrt_not_box:
                item_names.append('lrt_traj_camR')
            else:
                item_names.append('box_traj_camR')
                item_names.append('full_boxlist_camR')

        elif self.data_format=='oldtraj':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'box_traj_camR',
                'score_traj',
            ]
        elif self.data_format=='simpletraj':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camX0',
                'xyz_camX0',
                'box_traj_camR',
                'score_traj',
            ]
        elif self.data_format=='cdtb':
            item_names = [
                'pix_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'full_boxlist_2d_s',
                'depths'
            ]
        elif self.data_format=='ktrack':
            item_names = [
                'rgb_camXs',
                'xyz_veloXs',
                'origin_T_camXs',
                'pix_T_cams',
                'cams_T_velos',
                'boxlists',
                'tidlists',
                'scorelists',
            ]
        elif self.data_format=='nuscenes':
            item_names = [
                'rgb_camXs',
                'xyz_camXs',
                'origin_T_camXs',
                'pix_T_cams',
                'lrtlist_camXs',
                'scorelist_camXs',
                'tidlist_camXs',
                'lrtlist_camXs_det',
                'scorelist_camXs_det',
            ]
        elif self.data_format=='kodo':
            item_names = [
                'rgb_camXs',
                'xyz_veloXs',
                'origin_T_camXs',
                'pix_T_cams',
                'cams_T_velos',
                # 'boxlists',
                # 'tidlists',
                # 'scorelists',
            ]
        elif self.data_format=="3dq":
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
            ]
        else:
            item_names = None
        return item_names

    def random_select_single(self, batch, num_samples=2):
        item_names = self.get_item_names()
        # num_all = len(batch[item_names[origin]]) # total number of frames

        if self.data_format=='cater_traj' or self.data_format=='cater_traj_w_shape' or self.data_format=='cdtb':
            num_all = len(batch['pix_T_camXs']) # total number of frames
        elif self.data_format=='complete':
            num_all = len(batch['origin_T_cam0s']) # total number of frames
        elif self.data_format=='3dq':
            num_all = len(batch['origin_T_camXs_raw']) # total number of frames
        else:
            num_all = len(batch['origin_T_camXs']) # total number of frames

        if (self.data_format=='traj') or (self.data_format=='simpletraj') or (self.data_format=='ktrack') \
        or (self.data_format=='oldtraj') or (self.data_format=='kodo') or (self.data_format=='complete') \
        or (self.data_format=='intphys_fullseq') \
        or (self.data_format=='surveil_traj') or (self.data_format=='cater_traj') or (self.data_format=='cater_traj_w_shape'):
            # print('loading a traj')
            if self.data_consec:
                # we want a contiguous subseq

                # print('num_all', num_all)
                # print('num_samples', num_samples)
                
                stride = 1
                # print('gathering data with stride %d' % sptride)
                
                start_inds = list(range(num_all-num_samples))
                if num_all==num_samples:
                    start_ind = 0
                else:
                    start_ind = np.random.randint(0, num_all-num_samples*stride, 1).squeeze()
                # print('starting at %d' % start_ind)

                batch_new = {}
                inds = list(range(start_ind,start_ind+num_samples*stride,stride))
                for item_name in item_names:
                    if not (item_name in ['rgb_camX0', 'xyz_camX0']):
                        item = batch[item_name]
                        # item = item[start_ind:start_ind+num_samples]
                        item = item[inds]
                        batch_new[item_name] = item
                    else:
                        # copy directly
                        batch_new[item_name] = batch[item_name]
            else:
                inds = np.random.randint(0, num_all, num_samples)
                # print('setting ind0 to 0')
                # inds[0] = 0
                # print('setting ind1 to 1')
                # inds[1] = 1
                
                # print('taking inds', inds)

                batch_new = {}
                for item_name in item_names:
                    if not (item_name in ['rgb_camX0', 'xyz_camX0']):
                        item = batch[item_name]
                        item = item[inds]
                        batch_new[item_name] = item
                    else:
                        # copy directly
                        batch_new[item_name] = batch[item_name]
                
        else:
            # first shuffle
            inds = np.random.permutation(list(range(num_all)))
            inds = inds[:num_samples]
            
            # print('setting ind0 to 0')
            # inds[0] = 0
            
            batch_new = {}
            for item_name in item_names:
                item = batch[item_name]

                # print('item', item_name, len(item), item[0].shape)
                
                # item = item[inds]
                # # now select a random set of length num_samples
                # item = item[:num_samples]
                item = item[inds]
                
                batch_new[item_name] = item

        batch_new['ind_along_S'] = inds
        return batch_new

    def non_random_select_single(self, batch, num_samples=2):
        item_names = self.get_item_names()
        # print(item_names)
        # print(len(batch[item_names[0]]))
        # print(batch[item_names[0]].shape)
        # print(batch[item_names[1]].shape)

        # num_all = len(batch[item_names[0]]) # total number of frames
        # st()
        if 'origin_T_camXs' in batch.keys():
            num_all = len(batch['origin_T_camXs']) # total number of frames
        elif self.data_format=='complete':
            num_all = len(batch['origin_T_cam0s']) # total number of frames            
        elif self.data_format=='cdtb':
            num_all = len(batch['pix_T_camXs'])
        elif self.data_format=='3dq':
            num_all = len(batch['origin_T_camXs_raw']) # total number of frames
        else:
            num_all = len(batch['world_T_camXs']) # total number of frames

        # else:
        #     for k, v in batch.items():
        #         print(k, v.shape)
        #     # num_all = len(batch['vehicle_T_cams']) # total number of frames

        # print('num_all', num_all)

        batch_new = {}
        # select valid candidate
        if 'valid_pairs' in batch:
            valid_pairs = batch['valid_pairs'] #this is ? x 2
            sample_pair = -1
            sample_id = valid_pairs[sample_pair, :] #this is length-2
        else:
            sample_id = range(num_all)
            # sample_id = [0,1]
            # sample_id = [1,2]
            # sample_id = [2,3]
            # sample_id = [3,4]
            # sample_id = [4,5]
            # sample_id = [5,6]

        if len(sample_id) > num_samples:
            # final_sample = sample_id[-num_samples:]
            final_sample = sample_id[:num_samples]
        else:
            final_sample = sample_id
        # print('sample_id', sample_id)
        # print('final_sample', final_sample)

        if num_samples > len(sample_id):
            print('inputs.py warning: S larger than valid frames number')
            print('num_samples = %d;, len sample = %d;' % (num_samples, len(sample_id)))

        for item_name in item_names:
            # print('item_name', item_name)
            # print('item', batch[item_name].shape)
            if not (item_name in ['rgb_camX0', 'xyz_camX0']):
                item = batch[item_name]
                item = item[final_sample]
                batch_new[item_name] = item
            else:
                # copy directly
                batch_new[item_name] = batch[item_name]

        batch_new['ind_along_S'] = final_sample
        # else:
        #     batch_new = batch
        return batch_new

    def random_time_flip_single(self, sample):
        do_flip = np.random.randint(2) # 0 or 1
        item_names = self.get_item_names()
        for item_name in item_names:
            # print('flipping', item_name)
            item = sample[item_name]
            if do_flip > 0.5:
                # flip along the seq dim, which is 0
                if torch.is_tensor(item):
                    item = item.flip(0)
                else:
                    item = np.flip(item, axis=0).copy()
            sample[item_name] = item
        return sample

def get_inputs():
    dataset_filetype = hyp.dataset_filetype
    all_set_inputs = {}
    for set_name in hyp.set_names:
        if hyp.sets_to_run[set_name]:
            data_path = hyp.data_paths[set_name]
            shuffle = hyp.shuffles[set_name]
            data_format = hyp.data_formats[set_name]
            data_consec = hyp.data_consecs[set_name]
            seqlen = hyp.seqlens[set_name]
            batch_size = hyp.batch_sizes[set_name]
            if dataset_filetype == 'npz':
                # print('setting num_workers=4')
                # print('setting num_workers=1')
                if hyp.debug:
                    num_workers = 0
                else:
                    num_workers = 4
                # num_workers = 6
                # num_workers = 1
                # num_workers = 2
                print('setting num_workers=%d' % num_workers)
                
                dataset = IndexedDataset(NpzDataset(
                    dataset_path=data_path,
                    shuffle=shuffle,
                    data_format=data_format,
                    data_consec=data_consec,
                    seqlen=seqlen))
                # all_datasets[set_name] = dataset
                all_set_inputs[set_name] = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True)
            else:
                assert False # other filetypes not ready right now

    return all_set_inputs
