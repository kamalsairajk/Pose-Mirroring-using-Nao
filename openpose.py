import cv2
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from entity import params, JointType
from models.CocoPoseNet import CocoPoseNet, compute_loss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from datetime import datetime
import time
from matplotlib import pyplot as plt

def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

class Openpose(object):
    def __init__(self, arch='posenet', weights_file=None, training = True):
        self.arch = arch
        if weights_file:
            self.model = params['archs'][arch]()
            self.model.load_state_dict(torch.load(weights_file))
        else:
            self.model = params['archs'][arch](params['pretrained_path'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if training:
            from pycocotools.coco import COCO
            from coco_dataset import CocoDataset
            for para in self.model.base.vgg_base.parameters():
                para.requires_grad = False
            coco_train = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_train2017.json'))
            coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
            self.train_loader = DataLoader(CocoDataset(coco_train, params['insize']), 
                                                                                params['batch_size'], 
                                                                                shuffle=True, 
                                                                                pin_memory=False,
                                                                                num_workers=params['num_workers'])
            self.val_loader = DataLoader(CocoDataset(coco_val, params['insize'], mode = 'val'), 
                                                                                params['batch_size'], 
                                                                                shuffle=False, 
                                                                                pin_memory=False,
                                                                                num_workers=params['num_workers'])
            self.train_length = len(self.train_loader)
            self.val_length = len(self.val_loader)
            self.step = 0
            self.writer = SummaryWriter(params['log_path'])
            self.board_loss_every = self.train_length // params['board_loss_interval']
            self.evaluate_every = self.train_length // params['eval_interval']
            self.board_pred_image_every = self.train_length // params['board_pred_image_interval']
            self.save_every = self.train_length // params['save_interval']
            self.optimizer = Adam([
                {'params' : [*self.model.parameters()][20:24], 'lr' : params['lr'] / 4},
                {'params' : [*self.model.parameters()][24:], 'lr' : params['lr']}])
            # test only codes
            # self.board_loss_every = 5
            # self.evaluate_every = 5
            # self.board_pred_image_every = 5
            # self.save_every = 5

    def board_scalars(self, key, loss, paf_log, heatmap_log):
        self.writer.add_scalar('{}_loss'.format(key), loss, self.step)
        for stage, (paf_loss, heatmap_loss) in enumerate(zip(paf_log, heatmap_log)):
            self.writer.add_scalar('{}_paf_loss_stage{}'.format(key, stage), paf_loss, self.step)
            self.writer.add_scalar('{}_heatmap_loss_stage{}'.format(key, stage), heatmap_loss, self.step)

    def evaluate(self, num = 50):
        self.model.eval()
        count = 0
        running_loss = 0.
        running_paf_log = 0.
        running_heatmap_log = 0.
        with torch.no_grad():
            for imgs, pafs, heatmaps, ignore_mask in iter(self.val_loader):
                imgs, pafs, heatmaps, ignore_mask = imgs.to(self.device), pafs.to(self.device), heatmaps.to(self.device), ignore_mask.to(self.device)
                pafs_ys, heatmaps_ys = self.model(imgs)
                total_loss, paf_loss_log, heatmap_loss_log = compute_loss(pafs_ys, heatmaps_ys, pafs, heatmaps, ignore_mask)
                running_loss += total_loss.item()
                running_paf_log += paf_loss_log
                running_heatmap_log += heatmap_loss_log
                count += 1
                if count >= num:
                    break
        return running_loss / num, running_paf_log / num, running_heatmap_log / num
    
    def save_state(self, val_loss, to_save_folder=False, model_only=False):
        if to_save_folder:
            save_path = params['work_space']/'save'
        else:
            save_path = params['work_space']/'model'
        time = get_time()
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_val_loss:{}_step:{}.pth'.format(time, val_loss, self.step)))
        if not model_only:
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_val_loss:{}_step:{}.pth'.format(time, val_loss, self.step)))

    def load_state(self, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = params['work_space']/'save'
        else:
            save_path = params['work_space']/'model'          
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        print('load model_{}'.format(fixed_str))
        if not model_only:
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
            print('load optimizer_{}'.format(fixed_str))

    def resume_training_load(self, from_save_folder=False):
        if from_save_folder:
            save_path = params['work_space']/'save'
        else:
            save_path = params['work_space']/'model'  
        sorted_files = sorted([*save_path.iterdir()],  key=lambda x: os.path.getmtime(x), reverse=True)
        seeking_flag = True
        index = 0
        while seeking_flag:
            if index > len(sorted_files) - 2:
                break
            file_a = sorted_files[index]
            file_b = sorted_files[index + 1]
            if file_a.name.startswith('model'):
                fix_str = file_a.name[6:]
                self.step = int(fix_str.split(':')[-1].split('.')[0]) + 1
                if file_b.name == ''.join(['optimizer', '_', fix_str]):                    
                    if self.step > 2000:
                        for para in self.model.base.vgg_base.parameters():
                            para.requires_grad = True
                        self.optimizer.add_param_group({'params' : [*self.model.base.vgg_base.parameters()], 'lr' : params['lr'] / 4})
                    self.load_state(fix_str, from_save_folder)               
                    print(self.optimizer)
                    return
                else:
                    index += 1
                    continue
            elif file_a.name.startswith('optimizer'):
                fix_str = file_a.name[10:]
                self.step = int(fix_str.split(':')[-1].split('.')[0]) + 1
                if file_b.name == ''.join(['model', '_', fix_str]):
                    if self.step > 2000:
                        for para in self.model.base.vgg_base.parameters():
                            para.requires_grad = True
                        self.optimizer.add_param_group({'params' : [*self.model.base.vgg_base.parameters()], 'lr' : params['lr'] / 4})
                    self.load_state(fix_str, from_save_folder)               
                    print(self.optimizer)
                    return
                else:
                    index += 1
                    continue
            else:
                index += 1
                continue
        print('no available files founded')
        return

    def find_lr(self,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=4.,
                num=None):
        if not num:
            num = len(self.train_loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, pafs, heatmaps, ignore_mask) in tqdm(enumerate(self.train_loader), total=num):

            imgs, pafs, heatmaps, ignore_mask = imgs.to(self.device), pafs.to(self.device), heatmaps.to(self.device), ignore_mask.to(self.device)
            self.optimizer.zero_grad()
            batch_num += 1  
            pafs_ys, heatmaps_ys = self.model(imgs)
            loss, _, _ = compute_loss(pafs_ys, heatmaps_ys, pafs, heatmaps, ignore_mask)
            
            self.optimizer.step()
        
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses        

    def lr_schedule(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10.
        print(self.optimizer)

    def train(self, resume = False):
        running_loss = 0.
        running_paf_log = 0.
        running_heatmap_log = 0.
        if resume:
            self.resume_training_load()
        for epoch in range(60):
            for imgs, pafs, heatmaps, ignore_mask in tqdm(iter(self.train_loader)):
                if self.step == 2000:
                    for para in self.model.base.vgg_base.parameters():
                        para.requires_grad = True
                    self.optimizer.add_param_group({'params' : [*self.model.base.vgg_base.parameters()], 'lr' : params['lr'] / 4})
                if self.step == 100000 or self.step == 200000:
                    self.lr_schedule()

                imgs, pafs, heatmaps, ignore_mask = imgs.to(self.device), pafs.to(self.device), heatmaps.to(self.device), ignore_mask.to(self.device)
                self.optimizer.zero_grad()
                pafs_ys, heatmaps_ys = self.model(imgs)
                total_loss, paf_loss_log, heatmap_loss_log = compute_loss(pafs_ys, heatmaps_ys, pafs, heatmaps, ignore_mask)
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()
                running_paf_log += paf_loss_log
                running_heatmap_log += heatmap_loss_log

                if (self.step  % self.board_loss_every == 0) & (self.step != 0):
                    self.board_scalars('train', 
                                        running_loss / self.board_loss_every, 
                                        running_paf_log / self.board_loss_every,
                                        running_heatmap_log / self.board_loss_every)
                    running_loss = 0.
                    running_paf_log = 0.
                    running_heatmap_log = 0.
                
                if (self.step  % self.evaluate_every == 0) & (self.step != 0):
                    val_loss, paf_loss_val_log, heatmap_loss_val_log = self.evaluate(num = params['eva_num'])
                    self.model.train()
                    self.board_scalars('val', val_loss, paf_loss_val_log, heatmap_loss_val_log)
                    
                if (self.step  % self.board_pred_image_every == 0) & (self.step != 0):
                    self.model.eval()
                    with torch.no_grad():
                        for i in range(20):
                            img_id = self.val_loader.dataset.imgIds[i]
                            img_path = os.path.join(params['coco_dir'], 'val2017', self.val_loader.dataset.coco.loadImgs([img_id])[0]['file_name'])
                            img = cv2.imread(img_path)
                            # inference
                            poses, _ = self.detect(img)
                            # draw and save image
                            img = draw_person_pose(img, poses)
                            img = torch.tensor(img.transpose(2,0,1))
                            self.writer.add_image('pred_image_{}'.format(i), img, global_step=self.step)
                    self.model.train()

                if (self.step  % self.save_every == 0) & (self.step != 0):
                    self.save_state(val_loss)
                
                self.step += 1
                if self.step > 300000:
                    break

    def pad_image(self, img, stride, pad_value):
        h, w, _ = img.shape

        pad = [0] * 2
        pad[0] = (stride - (h % stride)) % stride  # down
        pad[1] = (stride - (w % stride)) % stride  # right

        img_padded = np.zeros((h+pad[0], w+pad[1], 3), 'uint8') + pad_value
        img_padded[:h, :w, :] = img.copy()
        return img_padded, pad

    def compute_optimal_size(self, orig_img, img_size, stride=8):
        orig_img_h, orig_img_w, _ = orig_img.shape
        aspect = orig_img_h / orig_img_w
        if orig_img_h < orig_img_w:
            img_h = img_size
            img_w = np.round(img_size / aspect).astype(int)
            surplus = img_w % stride
            if surplus != 0:
                img_w += stride - surplus
        else:
            img_w = img_size
            img_h = np.round(img_size * aspect).astype(int)
            surplus = img_h % stride
            if surplus != 0:
                img_h += stride - surplus
        return (img_w, img_h)

    def compute_peaks_from_heatmaps(self, heatmaps):
        """all_peaks: shape = [N, 5], column = (jointtype, x, y, score, index)"""
        heatmaps = heatmaps[:-1]

        all_peaks = []
        peak_counter = 0
        for i , heatmap in enumerate(heatmaps):
            heatmap = gaussian_filter(heatmap, sigma=params['gaussian_sigma'])
            
            map_left = np.zeros(heatmap.shape)
            map_right = np.zeros(heatmap.shape)
            map_top = np.zeros(heatmap.shape)
            map_bottom = np.zeros(heatmap.shape)
            
            map_left[1:, :] = heatmap[:-1, :]
            map_right[:-1, :] = heatmap[1:, :]
            map_top[:, 1:] = heatmap[:, :-1]
            map_bottom[:, :-1] = heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce((
                heatmap > params['heatmap_peak_thresh'],
                heatmap > map_left,
                heatmap > map_right,
                heatmap > map_top,
                heatmap > map_bottom,
            ))
            
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) 
            
            peaks_with_score = [(i,) + peak_pos + (heatmap[peak_pos[1], peak_pos[0]],) for peak_pos in peaks]
            
            peaks_id = range(peak_counter, peak_counter + len(peaks_with_score))
            peaks_with_score_and_id = [peaks_with_score[i] + (peaks_id[i], ) for i in range(len(peaks_id))]
            
            peak_counter += len(peaks_with_score_and_id)
            all_peaks.append(peaks_with_score_and_id)
        all_peaks = np.array([peak for peaks_each_category in all_peaks for peak in peaks_each_category])
        
        return all_peaks

    def compute_candidate_connections(self, paf, cand_a, cand_b, img_len, params):
        candidate_connections = []
        for joint_a in cand_a:
            for joint_b in cand_b:  # jointは(x, y)座標
                vector = joint_b[:2] - joint_a[:2]
                norm = np.linalg.norm(vector)
                if norm == 0:
                    continue
                ys = np.linspace(joint_a[1], joint_b[1], num=params['n_integ_points'])
                xs = np.linspace(joint_a[0], joint_b[0], num=params['n_integ_points'])
                integ_points = np.stack([ys, xs]).T.round().astype('i')  
                
                paf_in_edge = np.hstack([paf[0][np.hsplit(integ_points, 2)], paf[1][np.hsplit(integ_points, 2)]])
                
                unit_vector = vector / norm
                inner_products = np.dot(paf_in_edge, unit_vector)
                integ_value = inner_products.sum() / len(inner_products)
                
                integ_value_with_dist_prior = integ_value + min(params['limb_length_ratio'] * img_len / norm - params['length_penalty_value'], 0)
                
                n_valid_points = sum(inner_products > params['inner_product_thresh'])
                if n_valid_points > params['n_integ_points_thresh'] and integ_value_with_dist_prior > 0:
                    candidate_connections.append([int(joint_a[3]), int(joint_b[3]), integ_value_with_dist_prior])
                    
        candidate_connections = sorted(candidate_connections, key=lambda x: x[2], reverse=True)
        
        return candidate_connections
        

    def compute_connections(self, pafs, all_peaks, img_len, params):
        all_connections = []
        for i in range(len(params['limbs_point'])):
            
            paf_index = [i*2, i*2 + 1]
            paf = pafs[paf_index] 
            
            limb_point = params['limbs_point'][i] # example: [<JointType.Neck: 1>, <JointType.RightWaist: 8>]
            cand_a = all_peaks[all_peaks[:, 0] == limb_point[0]][:, 1:]
            cand_b = all_peaks[all_peaks[:, 0] == limb_point[1]][:, 1:]
            

            if len(cand_a) > 0 and len(cand_b) > 0:
                candidate_connections = self.compute_candidate_connections(paf, cand_a, cand_b, img_len, params)
                
                connections = np.zeros((0, 3))
                
                for index_a, index_b, score in candidate_connections:
                    if index_a not in connections[:, 0] and index_b not in connections[:, 1]:
                        
                        connections = np.vstack([connections, [index_a, index_b, score]])
                        if len(connections) >= min(len(cand_a), len(cand_b)):
                            break
                all_connections.append(connections)
            else:
                all_connections.append(np.zeros((0, 3)))
        return all_connections

    def grouping_key_points(self, all_connections, candidate_peaks, params):
        subsets = -1 * np.ones((0, 20))
        

        for l, connections in enumerate(all_connections):
            joint_a, joint_b = params['limbs_point'][l]
            
            for ind_a, ind_b, score in connections[:, :3]:
                
                ind_a, ind_b = int(ind_a), int(ind_b)
                joint_found_cnt = 0
                joint_found_subset_index = [-1, -1]
                for subset_ind, subset in enumerate(subsets):
                    
                    if subset[joint_a] == ind_a or subset[joint_b] == ind_b:
                        joint_found_subset_index[joint_found_cnt] = subset_ind
                        joint_found_cnt += 1
                
                if joint_found_cnt == 1: 
                                 
                    found_subset = subsets[joint_found_subset_index[0]]
                    
                    if found_subset[joint_b] != ind_b:
                        found_subset[joint_b] = ind_b
                        found_subset[-1] += 1 # increment joint count
                        found_subset[-2] += candidate_peaks[ind_b, 3] + score  

                elif joint_found_cnt == 2: 
                    found_subset_1 = subsets[joint_found_subset_index[0]]
                    found_subset_2 = subsets[joint_found_subset_index[1]]

                    membership = ((found_subset_1 >= 0).astype(int) + (found_subset_2 >= 0).astype(int))[:-2]
                    if not np.any(membership == 2):  # merge two subsets when no duplication
                        found_subset_1[:-2] += found_subset_2[:-2] + 1 # default is -1
                        found_subset_1[-2:] += found_subset_2[-2:]
                        found_subset_1[-2:] += score  
                        subsets = np.delete(subsets, joint_found_subset_index[1], axis=0)
                    else:
                        if found_subset_1[joint_a] == -1:
                            found_subset_1[joint_a] = ind_a
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_1[joint_b] == -1:
                            found_subset_1[joint_b] = ind_b
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_b, 3] + score
                        if found_subset_2[joint_a] == -1:
                            found_subset_2[joint_a] = ind_a
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_2[joint_b] == -1:
                            found_subset_2[joint_b] = ind_b
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_b, 3] + score

                elif joint_found_cnt == 0 and l != 9 and l != 13: 
                    
                    row = -1 * np.ones(20)
                    row[joint_a] = ind_a
                    row[joint_b] = ind_b
                    row[-1] = 2
                    row[-2] = sum(candidate_peaks[[ind_a, ind_b], 3]) + score
                    subsets = np.vstack([subsets, row])
                elif joint_found_cnt >= 3:
                    pass

        # delete low score subsets
        keep = np.logical_and(subsets[:, -1] >= params['n_subset_limbs_thresh'], subsets[:, -2]/subsets[:, -1] >= params['subset_score_thresh'])
        # params['n_subset_limbs_thresh'] = 3
        # params['subset_score_thresh'] = 0.2
        subsets = subsets[keep]
        return subsets


    def subsets_to_pose_array(self, subsets, all_peaks):
        
        person_pose_array = []
        for subset in subsets:
            joints = []
            for joint_index in subset[:18].astype('i'):
                if joint_index >= 0:
                    joint = all_peaks[joint_index][1:3].tolist()
                    joint.append(2)
                    joints.append(joint)
                else:
                    joints.append([0, 0, 0])
            person_pose_array.append(np.array(joints))
        person_pose_array = np.array(person_pose_array)
        return person_pose_array

    def compute_limbs_length(self, joints):
        limbs = []
        limbs_len = np.zeros(len(params["limbs_point"]))
        for i, joint_indices in enumerate(params["limbs_point"]):
            if joints[joint_indices[0]] is not None and joints[joint_indices[1]] is not None:
                limbs.append([joints[joint_indices[0]], joints[joint_indices[1]]])
                limbs_len[i] = np.linalg.norm(joints[joint_indices[1]][:-1] - joints[joint_indices[0]][:-1])
            else:
                limbs.append(None)

        return limbs_len, limbs

    def compute_unit_length(self, limbs_len):
        unit_length = 0
        base_limbs_len = limbs_len[[14, 3, 0, 13, 9]] 
        non_zero_limbs_len = base_limbs_len > 0
        if len(np.nonzero(non_zero_limbs_len)[0]) > 0:
            limbs_len_ratio = np.array([0.85, 2.2, 2.2, 0.85, 0.85])
            unit_length = np.sum(base_limbs_len[non_zero_limbs_len] / limbs_len_ratio[non_zero_limbs_len]) / len(np.nonzero(non_zero_limbs_len)[0])
        else:
            limbs_len_ratio = np.array([2.2, 1.7, 1.7, 2.2, 1.7, 1.7, 0.6, 0.93, 0.65, 0.85, 0.6, 0.93, 0.65, 0.85, 1, 0.2, 0.2, 0.25, 0.25])
            non_zero_limbs_len = limbs_len > 0
            unit_length = np.sum(limbs_len[non_zero_limbs_len] / limbs_len_ratio[non_zero_limbs_len]) / len(np.nonzero(non_zero_limbs_len)[0])

        return unit_length

    def get_unit_length(self, person_pose):
        limbs_length, limbs = self.compute_limbs_length(person_pose)
        unit_length = self.compute_unit_length(limbs_length)

        return unit_length

    def crop_around_keypoint(self, img, keypoint, crop_size):
        x, y = keypoint
        left = int(x - crop_size)
        top = int(y - crop_size)
        right = int(x + crop_size)
        bottom = int(y + crop_size)
        bbox = (left, top, right, bottom)

        cropped_img = self.crop_image(img, bbox)

        return cropped_img, bbox

    def crop_person(self, img, person_pose, unit_length):
        top_joint_priority = [4, 5, 6, 12, 16, 7, 13, 17, 8, 10, 14, 9, 11, 15, 2, 3, 0, 1, sys.maxsize]
        bottom_joint_priority = [9, 6, 7, 14, 16, 8, 15, 17, 4, 2, 0, 5, 3, 1, 10, 11, 12, 13, sys.maxsize]

        top_joint_index = len(top_joint_priority) - 1
        bottom_joint_index = len(bottom_joint_priority) - 1
        left_joint_index = 0
        right_joint_index = 0
        top_pos = sys.maxsize
        bottom_pos = 0
        left_pos = sys.maxsize
        right_pos = 0

        for i, joint in enumerate(person_pose):
            if joint[2] > 0:
                if top_joint_priority[i] < top_joint_priority[top_joint_index]:
                    top_joint_index = i
                elif bottom_joint_priority[i] < bottom_joint_priority[bottom_joint_index]:
                    bottom_joint_index = i
                if joint[1] < top_pos:
                    top_pos = joint[1]
                elif joint[1] > bottom_pos:
                    bottom_pos = joint[1]

                if joint[0] < left_pos:
                    left_pos = joint[0]
                    left_joint_index = i
                elif joint[0] > right_pos:
                    right_pos = joint[0]
                    right_joint_index = i

        top_padding_radio = [0.9, 1.9, 1.9, 2.9, 3.7, 1.9, 2.9, 3.7, 4.0, 5.5, 7.0, 4.0, 5.5, 7.0, 0.7, 0.8, 0.7, 0.8]
        bottom_padding_radio = [6.9, 5.9, 5.9, 4.9, 4.1, 5.9, 4.9, 4.1, 3.8, 2.3, 0.8, 3.8, 2.3, 0.8, 7.1, 7.0, 7.1, 7.0]

        left = (left_pos - 0.3 * unit_length).astype(int)
        right = (right_pos + 0.3 * unit_length).astype(int)
        top = (top_pos - top_padding_radio[top_joint_index] * unit_length).astype(int)
        bottom = (bottom_pos + bottom_padding_radio[bottom_joint_index] * unit_length).astype(int)
        bbox = (left, top, right, bottom)

        cropped_img = self.crop_image(img, bbox)
        return cropped_img, bbox

    def crop_face(self, img, person_pose, unit_length):
        face_size = unit_length
        face_img = None
        bbox = None

        # if have nose
        if person_pose[JointType.Nose][2] > 0:
            nose_pos = person_pose[JointType.Nose][:2]
            face_top = int(nose_pos[1] - face_size * 1.2)
            face_bottom = int(nose_pos[1] + face_size * 0.8)
            face_left = int(nose_pos[0] - face_size)
            face_right = int(nose_pos[0] + face_size)
            bbox = (face_left, face_top, face_right, face_bottom)
            face_img = self.crop_image(img, bbox)

        return face_img, bbox

    def crop_hands(self, img, person_pose, unit_length):
        hands = {
            "left": None,
            "right": None
        }

        if person_pose[JointType.LeftHand][2] > 0:
            crop_center = person_pose[JointType.LeftHand][:-1]
            if person_pose[JointType.LeftElbow][2] > 0:
                direction_vec = person_pose[JointType.LeftHand][:-1] - person_pose[JointType.LeftElbow][:-1]
                crop_center += (0.3 * direction_vec).astype(crop_center.dtype)
            hand_img, bbox = self.crop_around_keypoint(img, crop_center, unit_length * 0.95)
            hands["left"] = {
                "img": hand_img,
                "bbox": bbox
            }

        if person_pose[JointType.RightHand][2] > 0:
            crop_center = person_pose[JointType.RightHand][:-1]
            if person_pose[JointType.RightElbow][2] > 0:
                direction_vec = person_pose[JointType.RightHand][:-1] - person_pose[JointType.RightElbow][:-1]
                crop_center += (0.3 * direction_vec).astype(crop_center.dtype)
            hand_img, bbox = self.crop_around_keypoint(img, crop_center, unit_length * 0.95)
            hands["right"] = {
                "img": hand_img,
                "bbox": bbox
            }

        return hands

    def crop_image(self, img, bbox):
        left, top, right, bottom = bbox
        img_h, img_w, img_ch = img.shape
        box_h = bottom - top
        box_w = right - left

        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(img_w, right)
        crop_bottom = min(img_h, bottom)
        crop_h = crop_bottom - crop_top
        crop_w = crop_right - crop_left
        cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]

        bias_x = bias_y = 0
        if left < crop_left:
            bias_x = crop_left - left
        if top < crop_top:
            bias_y = crop_top - top

        # pad
        padded_img = np.zeros((box_h, box_w, img_ch), dtype=np.uint8)
        padded_img[bias_y:bias_y+crop_h, bias_x:bias_x+crop_w] = cropped_img
        return padded_img

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)[None]
        return x_data

    def detect_precise(self, orig_img):
        orig_img_h, orig_img_w, _ = orig_img.shape

        pafs_sum = 0
        heatmaps_sum = 0

        interpolation = cv2.INTER_CUBIC

        for scale in params['inference_scales']:
            # TTA, multl scale testing, scale in [0.5, 1, 1.5, 2]
            multiplier = scale * params['inference_img_size'] / min(orig_img.shape[:2])
            
            img = cv2.resize(orig_img, (math.ceil(orig_img_w*multiplier), math.ceil(orig_img_h*multiplier)), interpolation=interpolation)
            
            padded_img, pad = self.pad_image(img, params['downscale'], (104, 117, 123))
            

            x_data = self.preprocess(padded_img) 
            x_data = torch.tensor(x_data).to(self.device)
            x_data.requires_grad = False

            with torch.no_grad():

                h1s, h2s = self.model(x_data) 

            tmp_paf = h1s[-1][0].cpu().numpy().transpose(1, 2, 0)
            tmp_heatmap = h2s[-1][0].cpu().numpy().transpose(1, 2, 0)

            p_h, p_w = padded_img.shape[:2]
            tmp_paf = cv2.resize(tmp_paf, (p_w, p_h), interpolation=interpolation)
            
            tmp_paf = tmp_paf[:p_h-pad[0], :p_w-pad[1], :]
            
            pafs_sum += cv2.resize(tmp_paf, (orig_img_w, orig_img_h), interpolation=interpolation)
            

            tmp_heatmap = cv2.resize(tmp_heatmap, (0, 0), fx=params['downscale'], fy=params['downscale'], interpolation=interpolation)
            tmp_heatmap = tmp_heatmap[:padded_img.shape[0]-pad[0], :padded_img.shape[1]-pad[1], :]
            heatmaps_sum += cv2.resize(tmp_heatmap, (orig_img_w, orig_img_h), interpolation=interpolation)
            

        self.pafs = (pafs_sum / len(params['inference_scales'])).transpose(2, 0, 1)
        self.heatmaps = (heatmaps_sum / len(params['inference_scales'])).transpose(2, 0, 1)

        self.all_peaks = self.compute_peaks_from_heatmaps(self.heatmaps)
        if len(self.all_peaks) == 0:
            return np.empty((0, len(JointType), 3)), np.empty(0)
        all_connections = self.compute_connections(self.pafs, self.all_peaks, orig_img_w, params)
        subsets = self.grouping_key_points(all_connections, self.all_peaks, params)
        poses = self.subsets_to_pose_array(subsets, self.all_peaks)
        scores = subsets[:, -2]
        return poses, scores

    def detect(self, orig_img, precise = False):
        orig_img = orig_img.copy()
        if precise:
            return self.detect_precise(orig_img)
        orig_img_h, orig_img_w, _ = orig_img.shape

        input_w, input_h = self.compute_optimal_size(orig_img, params['inference_img_size'])
        map_w, map_h = self.compute_optimal_size(orig_img, params['heatmap_size'])

        resized_image = cv2.resize(orig_img, (input_w, input_h))
        x_data = self.preprocess(resized_image)

        x_data = torch.tensor(x_data).to(self.device)
        x_data.requires_grad = False

        with torch.no_grad():

            h1s, h2s = self.model(x_data)

            pafs = F.interpolate(h1s[-1], (map_h, map_w), mode='bilinear', align_corners=True).cpu().numpy()[0]
            heatmaps = F.interpolate(h2s[-1], (map_h, map_w), mode='bilinear', align_corners=True).cpu().numpy()[0]

        all_peaks = self.compute_peaks_from_heatmaps(heatmaps)
        if len(all_peaks) == 0:
            return np.empty((0, len(JointType), 3)), np.empty(0)
        all_connections = self.compute_connections(pafs, all_peaks, map_w, params)
        subsets = self.grouping_key_points(all_connections, all_peaks, params)
        all_peaks[:, 1] *= orig_img_w / map_w
        all_peaks[:, 2] *= orig_img_h / map_h
        poses = self.subsets_to_pose_array(subsets, all_peaks)
        scores = subsets[:, -2]
        return poses, scores


def draw_person_pose(orig_img, poses):
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    if len(poses) == 0:
        return orig_img

    limb_colors = [
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
        [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
        [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
        [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
    ]

    joint_colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    canvas = orig_img.copy()

    # limbs
    for pose in poses.round().astype('i'):
        for i, (limb, color) in enumerate(zip(params['limbs_point'], limb_colors)):
            if i != 9 and i != 13:  # don't show ear-shoulder connection
                limb_ind = np.array(limb)
                if np.all(pose[limb_ind][:, 2] != 0):
                    joint1, joint2 = pose[limb_ind][:, :2]
                    cv2.line(canvas, tuple(joint1), tuple(joint2), color, 2)

    # joints
    for pose in poses.round().astype('i'):
        for i, ((x, y, v), color) in enumerate(zip(pose, joint_colors)):
            if v != 0:
                cv2.circle(canvas, (x, y), 3, color, -1)
    return canvas
