import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from .registry import DATASETS
from .process import Process
from clrnet.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC
from tools import reshape    ### Punit added

@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split
        self.processes = Process(processes, cfg)
        self.get_reshape()
    
    def get_reshape(self):
        real_img_siz = [self.cfg.real_img_h,self.cfg.real_img_w]
        exp_img_siz = [self.cfg.ori_img_h,self.cfg.ori_img_w]
        
        self.resize = reshape.reshape(real_img_siz,exp_img_siz)

    def view(self, predictions,confidences, img_metas):
        
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        
        for lanes,confidence, img_meta in zip(predictions,confidences, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            
            out_file = osp.join(self.cfg.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [self.resize.data(lane.to_array(self.cfg)) for lane in lanes]     ### Punit edits
            #lanes = [lane.to_array(self.cfg) for lane in lanes]     ### Punit edits
            #img = self.resize.image(img)
            #imshow_lanes(img, lanes, confidence,out_file=out_file)
            
            ## Punit additions to save text file of entries
            out_file = osp.join(self.cfg.work_dir,img_name.replace('/', '_'))
            out_file_strp = os.path.splitext(out_file)[0]
            
            with open(out_file_strp+'.txt','w') as lanes_txt:
                for i_lane,row in enumerate(lanes):
                    n_markers = len(row[:,1])
                    np.savetxt(lanes_txt,np.append(np.ones([n_markers,1])*i_lane,row,1),fmt='%d')

            conf_dir = os.path.dirname(out_file_strp)
            conf_file = os.path.basename(out_file_strp)
            with open(os.path.join(conf_dir,"conf_"+conf_file+".txt"),'w') as conf_txt:
                np.savetxt(conf_txt,confidence,delimiter=',', fmt='%1.3f')


            #with open(out_file_strp+'.txt','w') as lanes_txt:
                #for i_lane,xys in enumerate(lanes):
                    #print("base_dataset.py", out_file_strp, len(xys))
                    #if len(xys)>1:
                        #n_markers = len(xys[:,1])
                        #print(i_lane, n_markers, confidence)
                        #xysc = np.append(xys,np.ones([n_markers,1])*confidence[i_lane],1)
                        #np.savetxt(lanes_txt,np.append(np.ones([n_markers,1])*i_lane,xysc,1),fmt='%d %d %d %f')
                    
                    
                    
    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img_orig = cv2.imread(data_info['img_path'])
        img = img_orig[:,:,:3]
        img = self.resize.image(img)            ### Punit added
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample['lanes']:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({'lanes': new_lanes})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})

        return sample
