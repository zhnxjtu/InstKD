import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import common_utils



class KDPointTrans_point(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.point_cloud_range = point_cloud_range

        ''' Relation KD'''
        if self.model_cfg.RELA_KD.ENABLE:
            if self.model_cfg.RELA_KD.RELA_MODE == 'dir':
                self.feature_alignment = nn.Sequential(
                    nn.Linear(self.model_cfg.NUM_FEATURES, self.model_cfg.NUM_FEATURES*self.model_cfg.FACTOR, bias=False),
                    nn.BatchNorm1d(self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.model_cfg.NUM_FEATURES*self.model_cfg.FACTOR, self.model_cfg.NUM_FEATURES, bias=False),
                    nn.BatchNorm1d(self.model_cfg.NUM_FEATURES, eps=1e-3, momentum=0.01),
                )

            if self.model_cfg.RELA_KD.RELA_MODE == 'div':
                self.feature_alignment_inter = nn.Sequential(
                    nn.Linear(self.model_cfg.NUM_FEATURES, self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, bias=False),
                    nn.BatchNorm1d(self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, eps=1e-3, momentum=0.01),
                    nn.Linear(self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR,
                              bias=False),
                    nn.BatchNorm1d(self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, eps=1e-3, momentum=0.01),
                )
                self.feature_alignment_intri = nn.Sequential(
                    nn.Linear(self.model_cfg.NUM_FEATURES, self.model_cfg.NUM_FEATURES* self.model_cfg.FACTOR, bias=False),
                    nn.BatchNorm1d(self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, eps=1e-3, momentum=0.01),
                    nn.Linear(self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR,
                              bias=False),
                    nn.BatchNorm1d(self.model_cfg.NUM_FEATURES * self.model_cfg.FACTOR, eps=1e-3, momentum=0.01),
                )

        ''' feature-based KD'''
        self.feature_alignment_channel = nn.Sequential(
            nn.Linear(self.model_cfg.NUM_OUTPUT_FEATURES, self.model_cfg.NUM_OUTPUT_STUDENT, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_STUDENT, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        ''' Instance-aware KD'''
        if self.model_cfg.INSTANCE_WEIGHT.ENABLE:
            self.sigmoid = nn.Sigmoid()

    @staticmethod
    def reorder_feature(stu_feature, tea_feature):

        cur_coords = stu_feature.indices.clone()
        cur_features = stu_feature.features.contiguous().clone()

        cur_coords_tea = tea_feature.indices.clone()
        cur_features_tea = tea_feature.features.contiguous().clone()

        _, idx1 = cur_coords[:, 0].sort(dim=0)
        cur_coords1 = cur_coords[idx1]
        cur_features1 = cur_features[idx1]

        _, idx2 = cur_coords1[:, 1].sort(dim=0)
        cur_coords2 = cur_coords1[idx2]
        cur_features2 = cur_features1[idx2]

        _, idx3 = cur_coords2[:, 2].sort(dim=0)
        cur_coords3 = cur_coords2[idx3]
        cur_features3 = cur_features2[idx3]

        _, idx4 = cur_coords3[:, 3].sort(dim=0)
        cur_coords4 = cur_coords3[idx4]
        cur_features4 = cur_features3[idx4]

        ''' Teacher '''
        _, idx_t1 = cur_coords_tea[:, 0].sort(dim=0)
        cur_coords_tea1 = cur_coords_tea[idx_t1]
        cur_features_tea1 = cur_features_tea[idx_t1]

        _, idx_t2 = cur_coords_tea1[:, 1].sort(dim=0)
        cur_coords_tea2 = cur_coords_tea1[idx_t2]
        cur_features_tea2 = cur_features_tea1[idx_t2]

        _, idx_t3 = cur_coords_tea2[:, 2].sort(dim=0)
        cur_coords_tea3 = cur_coords_tea2[idx_t3]
        cur_features_tea3 = cur_features_tea2[idx_t3]

        _, idx_t4 = cur_coords_tea3[:, 3].sort(dim=0)
        cur_coords_tea4 = cur_coords_tea3[idx_t4]
        cur_features_tea4 = cur_features_tea3[idx_t4]

        stu_coords = cur_coords4
        stu_features = cur_features4
        tea_coords = cur_coords_tea4
        tea_features = cur_features_tea4

        return stu_coords, stu_features, tea_coords, tea_features

    def roi2voxel(self, rois, batch_size):
        with torch.no_grad():
            label = rois[:, :, 7].unsqueeze(-1)

            rois = rois.view(-1, rois.shape[-1])

            xyz = rois[:, 0:3].clone()
            lhw = rois[:, 3:6] / 2
            angle = rois[:, 6]

            coord_min = xyz - lhw - xyz
            x_min = coord_min[:, 0].unsqueeze(-1)
            y_min = coord_min[:, 1].unsqueeze(-1)
            z_min = coord_min[:, 2].unsqueeze(-1)
            coord_max = xyz + lhw - xyz
            x_max = coord_max[:, 0].unsqueeze(-1)
            y_max = coord_max[:, 1].unsqueeze(-1)
            z_max = coord_max[:, 2].unsqueeze(-1)
            # print(coord_min[0])
            # assert False

            coord1 = coord_min.unsqueeze(1)
            coord2 = torch.cat((x_max, y_min, z_min), dim=-1).unsqueeze(1)
            coord3 = torch.cat((x_min, y_max, z_min), dim=-1).unsqueeze(1)
            coord4 = torch.cat((x_max, y_max, z_min), dim=-1).unsqueeze(1)
            coord5 = torch.cat((x_max, y_min, z_max), dim=-1).unsqueeze(1)
            coord6 = torch.cat((x_min, y_max, z_max), dim=-1).unsqueeze(1)
            coord7 = torch.cat((x_min, y_min, z_max), dim=-1).unsqueeze(1)
            coord8 = coord_max.unsqueeze(1)

            coord = torch.cat((coord1, coord2, coord3, coord4,
                               coord5, coord6, coord7, coord8), dim=1)

            global_roi_grid_points = common_utils.rotate_points_along_z(coord, angle).squeeze(dim=1)

            global_roi_grid_points = global_roi_grid_points + xyz.unsqueeze(dim=1)
            global_roi_grid_points_max = global_roi_grid_points.max(dim=1)[0]
            global_roi_grid_points_min = global_roi_grid_points.min(dim=1)[0]

            grid_xyz_max = global_roi_grid_points_max.view(batch_size, -1, 3)
            grid_xyz_min = global_roi_grid_points_min.view(batch_size, -1, 3)
            roi_grid_coords_max =  torch.cat([grid_xyz_max, label], dim=-1)
            roi_grid_coords_min =  torch.cat([grid_xyz_min, label], dim=-1)

        return roi_grid_coords_max, roi_grid_coords_min

    @staticmethod
    def KD_distance(logits_student, logits_teacher, temperature):
        log_pred_student = F.log_softmax(logits_student / temperature, dim=-1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=-1)
        KL = F.kl_div(log_pred_student, pred_teacher, reduction="none").mean(-1)

        return KL


    def forward(self, batch_dict):

        batch_dict['point_cls_preds'] = batch_dict['batch_cls_preds']

        if self.model_cfg.INSTANCE_WEIGHT.ENABLE:
            # b, c = batch_dict['point_cls_preds_tea'].shape
            weight = self.KD_distance(batch_dict['point_cls_preds'],
                                      batch_dict['point_cls_preds_tea'], 1.0)
            # weight = weight.reshape(b, h, w)

        rois = batch_dict['gt_boxes'].detach()
        batch_size = batch_dict['batch_size']

        stu_features = batch_dict['point_features']
        stu_coords = batch_dict['point_coords']
        with torch.no_grad():
            tea_features = batch_dict['point_features_tea']
            tea_coords = batch_dict['point_coords_tea']

        assert torch.equal(stu_coords, tea_coords)

        # print('a',stu_coords.max(0)[0])
        # print('b',stu_coords.min(0)[0])


        ''' FITNET for Channel alignment '''
        # stu_fea_trans = batch_dict['multi_scale_3d_features'][self.model_cfg.ALIGN_SOURCE]
        stu_features_trans = self.feature_alignment_channel(stu_features)
        roi_grid_coords_max, roi_grid_coords_min = self.roi2voxel(rois, batch_size) # (batch_size * anchor_num * (x_bound, y_bound, z_bound, label))

        cur_grid_max = roi_grid_coords_max.clone()
        cur_grid_min = roi_grid_coords_min.clone()
        # cur_grid_max = cur_grid_max[:, :, [2, 1, 0, 3]] # z, y, x
        # cur_grid_min = cur_grid_min[:, :, [2, 1, 0, 3]] # z, y, x


        cur_stu_fea_trans_list = []
        cur_tea_fea_list = []
        cur_pure_tea_fea_list = []
        cur_pure_stu_fea_list = []
        information_align_list = []
        stu_background = []
        tea_background = []
        for bs in range(batch_size):
            high_bound = cur_grid_max[bs]
            low_bound = cur_grid_min[bs]

            mask_ = torch.zeros((stu_features.shape[0]))
            mask_[torch.nonzero(stu_coords[:, 0] == bs).squeeze()] = 1

            for label_count in range(high_bound[:,3].max().int()):

                # find the object, 1 for car, 2 for pedestrian, 3for cyclist
                idx_label = torch.nonzero(high_bound[:,3] == (label_count+1)).squeeze()
                if idx_label.dim() == 0:
                    idx_label = idx_label.unsqueeze(0)

                cur_high_bound = high_bound[idx_label]
                cur_low_bound = low_bound[idx_label]

                num_anchor = cur_low_bound.shape[0]

                for anchor_count in range(num_anchor):

                    with torch.no_grad():
                        mask1 = torch.zeros((stu_features.shape[0]), dtype=torch.uint8)
                        mask2 = torch.zeros((stu_features.shape[0]), dtype=torch.uint8)
                        mask3 = torch.zeros((stu_features.shape[0]), dtype=torch.uint8)
                        mask4 = torch.zeros((stu_features.shape[0]), dtype=torch.uint8)
                        mask5 = torch.zeros((stu_features.shape[0]), dtype=torch.uint8)
                        mask6 = torch.zeros((stu_features.shape[0]), dtype=torch.uint8)
                        mask7 = torch.zeros((stu_features.shape[0]), dtype=torch.uint8)

                        if (stu_coords != tea_coords).sum().bool():
                            assert False

                        idx_cur_batch = torch.nonzero(stu_coords[:, 0] == bs).squeeze()
                        idx_max_x = torch.nonzero(stu_coords[:, 1] <= cur_high_bound[anchor_count, 0]).squeeze()
                        idx_min_x = torch.nonzero(stu_coords[:, 1] >= cur_low_bound[anchor_count, 0]).squeeze()
                        idx_max_y = torch.nonzero(stu_coords[:, 2] <= cur_high_bound[anchor_count, 1]).squeeze()
                        idx_min_y = torch.nonzero(stu_coords[:, 2] >= cur_low_bound[anchor_count, 1]).squeeze()
                        idx_max_z = torch.nonzero(stu_coords[:, 3] <= cur_high_bound[anchor_count, 2]).squeeze()
                        idx_min_z = torch.nonzero(stu_coords[:, 3] >= cur_low_bound[anchor_count, 2]).squeeze()

                        mask1[idx_cur_batch] = 1
                        mask2[idx_max_z] = 1
                        mask3[idx_min_z] = 1
                        mask4[idx_max_y] = 1
                        mask5[idx_min_y] = 1
                        mask6[idx_max_x] = 1
                        mask7[idx_min_x] = 1
                        mask = mask1*mask2*mask3*mask4*mask5*mask6*mask7

                        idx_obj = torch.nonzero(mask).squeeze()
                        if idx_obj.dim() == 0:
                            idx_obj = idx_obj.unsqueeze(0)

                    cur_stu_fea = stu_features[idx_obj]
                    cur_tea_fea = tea_features[idx_obj]
                    cur_stu_fea_trans = stu_features_trans[idx_obj]

                    # bach, label, anchor_count, num_point, channel_num
                    information_align = torch.zeros((cur_tea_fea.shape[0], 4))
                    information_align[:, 0] = bs
                    information_align[:, 1] = label_count
                    information_align[:, 2] = anchor_count
                    information_align[:, 3] = cur_tea_fea.shape[0]

                    if self.model_cfg.RELA_KD.ENABLE:
                        # cur_pure_stu_fea = cur_stu_fea.clone()
                        cur_pure_stu_fea = cur_stu_fea_trans.clone()
                        cur_pure_tea_fea = cur_tea_fea.clone()
                        cur_pure_tea_fea_list.append(cur_pure_tea_fea)
                        cur_pure_stu_fea_list.append(cur_pure_stu_fea)

                    if self.model_cfg.INSTANCE_WEIGHT.ENABLE:
                        # mask_coords = stu_coords[idx_obj, 2:4].clone()
                        # mask_coords = torch.unique(mask_coords, dim=0)
                        if idx_obj.numel():
                            # cur_weight = weight[bs, stu_coords[idx_mask, 2].long(), stu_coords[idx_mask, 3].long()].mean()
                            cur_weight = weight[idx_obj].mean()
                            cur_weight = self.sigmoid(self.model_cfg.INSTANCE_WEIGHT.WEIGHT * cur_weight)
                            cur_stu_fea_trans = cur_weight * cur_stu_fea_trans
                            # print(cur_weight)
                            cur_tea_fea = cur_weight * cur_tea_fea

                    cur_tea_fea_list.append(cur_tea_fea)
                    cur_stu_fea_trans_list.append(cur_stu_fea_trans)
                    information_align_list.append(information_align)

                    ''' Background '''
                    mask_ = mask_ * (1 - mask)

            idx_back = torch.nonzero(mask_).squeeze()
            if idx_back.dim() == 0:
                idx_back = idx_back.unsqueeze(0)
            stu_background.append(stu_features_trans[idx_back])
            tea_background.append(tea_features[idx_back])

        information_align_list_total = torch.cat(information_align_list, dim=0)
        cur_stu_trans_list_total = torch.cat(cur_stu_fea_trans_list, dim=0)
        cur_tea_list_total = torch.cat(cur_tea_fea_list, dim=0)
        stu_background_total = torch.cat(stu_background, dim=0)
        tea_background_total = torch.cat(tea_background, dim=0)
        if self.model_cfg.RELA_KD.ENABLE:
            cur_pure_tea_list_total = torch.cat(cur_pure_tea_fea_list, dim=0)
            cur_pure_stu_list_total = torch.cat(cur_pure_stu_fea_list, dim=0)

        if self.model_cfg.FEA_KD.MODE == 'dir':
            batch_dict['voxel_features_tea_global'] = tea_features
            batch_dict['voxel_features_stu_global'] = stu_features_trans
        elif self.model_cfg.FEA_KD.MODE == 'div':
            batch_dict['voxel_features_tea_fg'] = cur_tea_list_total  # (BxN, C)
            batch_dict['voxel_features_stu_fg'] = cur_stu_trans_list_total
            batch_dict['voxel_features_tea_bg'] = tea_background_total
            batch_dict['voxel_features_stu_bg'] = stu_background_total

        ''' Relation KD '''
        if self.model_cfg.RELA_KD.ENABLE:
            max_num_point = information_align_list_total[:, 3].max().int()
            max_anchor_num = (information_align_list_total[:, 2].max() + 1).int()
            max_label = (information_align_list_total[:, 1].max() + 1).int()
            stu_matrix = cur_pure_stu_list_total.\
                new_zeros((batch_size, max_label, max_anchor_num, max_num_point, cur_pure_stu_list_total.shape[1]))
            # with torch.no_grad():
            tea_matrix = cur_pure_tea_list_total.\
                new_zeros((batch_size, max_label, max_anchor_num, max_num_point, cur_pure_tea_list_total.shape[1]))

            for bs in range(batch_size):
                for label_count in range(max_label):
                    for anchor_count in range(max_anchor_num):

                        with torch.no_grad():
                            mask1 = torch.zeros((cur_pure_stu_list_total.shape[0]), dtype=torch.uint8)
                            mask2 = torch.zeros((cur_pure_stu_list_total.shape[0]), dtype=torch.uint8)
                            mask3 = torch.zeros((cur_pure_stu_list_total.shape[0]), dtype=torch.uint8)

                            idx_bs = torch.nonzero(information_align_list_total[:, 0] == bs).squeeze()
                            idx_label = torch.nonzero(information_align_list_total[:, 1] == label_count).squeeze()
                            idx_anchor = torch.nonzero(information_align_list_total[:, 2] == anchor_count).squeeze()
                            if idx_label.dim() ==0:
                                idx_label = idx_label.unsqueeze(0)
                            if idx_anchor.dim() == 0:
                                idx_anchor = idx_anchor.unsqueeze(0)

                            mask1[idx_bs] = 1
                            mask2[idx_label] = 1
                            mask3[idx_anchor] = 1

                            mask = mask1*mask2*mask3
                            idx = torch.nonzero(mask).squeeze()

                            if idx.dim() ==0:
                                idx = idx.unsqueeze(0)

                            if (idx.numel() != 0):
                                if len(idx) != information_align_list_total[idx[0], 3]:
                                    assert False

                        stu_matrix[bs, label_count, anchor_count, 0:len(idx), :] = cur_pure_stu_list_total[idx]
                        # with torch.no_grad():
                        tea_matrix[bs, label_count, anchor_count, 0:len(idx), :] = cur_pure_tea_list_total[idx]


            intri_stu_total_list = []
            inter_stu_total_list = []
            intri_tea_total_list = []
            inter_tea_total_list = []
            sample_num = self.model_cfg.NUM_FEATURES
            for bs in range(batch_size):

                intri_stu_list = []
                inter_stu_list = []
                intri_tea_list = []
                inter_tea_list = []
                cur_stu_matrix = stu_matrix[bs]
                cur_tea_matrix = tea_matrix[bs]

                for cur_label in range(max_label):
                    # teacher relation
                    # with torch.no_grad():
                    cur_label_tea_matrix = cur_tea_matrix[cur_label][:,0:sample_num,:]
                    idx_keep = torch.tensor([i for i in range(max_label) if i != cur_label])
                    cur_other_tea_matrix = torch.index_select(cur_tea_matrix, 0, idx_keep.cuda())[:,0:sample_num,:]

                    cur_label_tea_matrix_resize = cur_label_tea_matrix.reshape(-1, cur_tea_list_total.shape[1]).unsqueeze(
                        0).repeat(max_anchor_num, 1, 1)
                    cur_other_tea_matrix_resize = cur_other_tea_matrix.reshape(-1, cur_tea_list_total.shape[1]).unsqueeze(
                        0).repeat(max_anchor_num, 1, 1)

                    intri_tea = torch.bmm(cur_label_tea_matrix, cur_label_tea_matrix_resize.permute(0, 2, 1)) / \
                                cur_tea_list_total.shape[1]
                    inter_tea = torch.bmm(cur_label_tea_matrix, cur_other_tea_matrix_resize.permute(0, 2, 1)) / \
                                cur_tea_list_total.shape[1]

                    intri_tea = intri_tea.permute(0, 2, 1).reshape(-1, sample_num)
                    inter_tea = inter_tea.permute(0, 2, 1).reshape(-1, sample_num)
                    # print(inter_tea.shape)
                    # exit()

                    idx_nonzero_intri = torch.nonzero(intri_tea.sum(-1)).squeeze()
                    idx_nonzero_inter = torch.nonzero(inter_tea.sum(-1)).squeeze()
                    intri_tea = intri_tea[idx_nonzero_intri]
                    inter_tea = inter_tea[idx_nonzero_inter]

                    intri_tea_list.append(intri_tea)
                    inter_tea_list.append(inter_tea)

                    # student relation
                    cur_label_stu_matrix = cur_stu_matrix[cur_label][:,0:sample_num,:]
                    cur_other_stu_matrix = torch.index_select(cur_stu_matrix, 0, idx_keep.cuda())[:,0:sample_num,:]

                    cur_label_stu_matrix_resize = cur_label_stu_matrix.reshape(-1, cur_pure_stu_list_total.shape[1]).unsqueeze(
                        0).repeat(max_anchor_num, 1, 1)
                    cur_other_stu_matrix_resize = cur_other_stu_matrix.reshape(-1, cur_pure_stu_list_total.shape[1]).unsqueeze(
                        0).repeat(max_anchor_num, 1, 1)

                    intri_stu = torch.bmm(cur_label_stu_matrix, cur_label_stu_matrix_resize.permute(0, 2, 1)) / \
                                cur_pure_stu_list_total.shape[1]
                    inter_stu = torch.bmm(cur_label_stu_matrix, cur_other_stu_matrix_resize.permute(0, 2, 1)) / \
                                cur_pure_stu_list_total.shape[1]

                    intri_stu = intri_stu.permute(0, 2, 1).reshape(-1, sample_num)
                    inter_stu = inter_stu.permute(0, 2, 1).reshape(-1, sample_num)

                    intri_stu = intri_stu[idx_nonzero_intri]
                    inter_stu = inter_stu[idx_nonzero_inter]

                    intri_stu_list.append(intri_stu)
                    inter_stu_list.append(inter_stu)


                intri_stu_total_list.append(torch.cat(intri_stu_list, dim=0))
                inter_stu_total_list.append(torch.cat(inter_stu_list, dim=0))
                # with torch.no_grad():
                intri_tea_total_list.append(torch.cat(intri_tea_list, dim=0))
                inter_tea_total_list.append(torch.cat(inter_tea_list, dim=0))

            rela_intri_stu_total = torch.cat(intri_stu_total_list, dim=0)
            rela_inter_stu_total = torch.cat(inter_stu_total_list, dim=0)
            # with torch.no_grad():
            rela_intri_tea_total = torch.cat(intri_tea_total_list, dim=0)
            rela_inter_tea_total = torch.cat(inter_tea_total_list, dim=0)


            if self.model_cfg.RELA_KD.RELA_MODE == 'div':
                rela_intri_stu_total = self.feature_alignment_intri(rela_intri_stu_total)
                rela_inter_stu_total = self.feature_alignment_inter(rela_inter_stu_total)
                batch_dict['rela_features_stu_intri'] = rela_intri_stu_total
                batch_dict['rela_features_stu_inter'] = rela_inter_stu_total
                # with torch.no_grad():
                batch_dict['rela_features_tea_intri'] = rela_intri_tea_total
                batch_dict['rela_features_tea_inter'] = rela_inter_tea_total

            if self.model_cfg.RELA_KD.RELA_MODE == 'dir':
                rela_stu_all = torch.cat((rela_intri_stu_total, rela_inter_stu_total), dim=0)
                rela_tea_all = torch.cat((rela_intri_tea_total, rela_inter_tea_total), dim=0)
                rela_stu_trans = self.feature_alignment(rela_stu_all)
                batch_dict['rela_features_tea'] = rela_tea_all  # (BxN, C)
                batch_dict['rela_features_stu'] = rela_stu_trans


        return batch_dict
