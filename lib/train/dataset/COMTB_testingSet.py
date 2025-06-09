import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class COMTB_testingSet(BaseVideoDataset):
    """
    RGBNIR 数据集:
        - groundtruth_rect.txt
        - rgb-nir.txt: 0-RGB 模态; 1-NIR 模态; 2-; 3-
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        self.root = env_settings().comtb_testingset_dir if root is None else root
        super().__init__('COMTB_testingSet', root, image_loader)

        # video_name for each sequence
        # NOTE: Easy_Set_Test
        self.easy_list = ['2', '6', '7', '8', '12', '19', '20', '21', '22', '26', '28', '29', '31', '32', '33',
                          '38', '40', '41', '43', '44', '45', '47', '50', '51', '55', '57', '58', '59', '60', '61',
                          '64', '67', '69', '70', '76', '80', '81', '84', '90', '92', '96', '98', '99', '100',
                          '101', '103', '107', '108', '109', '112', '114', '115', '116', '118', '119', '120', '122',
                          '124', '127', '128', '129', '131', '132', '135', '137', '139', '140', '141', '142', '143',
                          '145', '147', '150', '151', '153', '157', '159', '160', '162', '163', '166', '167', '174',
                          '176', '179', '188', '191', '192', '195', '197', '198', '202', '203', '204', '206', '209',
                          '210', '211', '214', '220', '222', '223', '226', '237', '243', '244', '245', '247', '249',
                          '251', '252', '254', '255', '256', '258', '260', '262', '263', '264', '266', '275', '277',
                          '278', '279', '281', '282', '284', '285', '287', '288', '290', '295', '296', '306', '307',
                          '320', '322', '323', '326', '332', '335', '336', '337', '338', '340', '341', '343', '347',
                          '348', '349', '350', '352', '356', '361', '367', '370', '373', '374', '376', '381', '382',
                          '384', '385', '386', '391', '394', '396', '397', '398', '400', '403', '405', '406', '408',
                          '409', '410', '411', '412', '414', '415', '416', '417', '419', '420', '423', '424', '426',
                          '427', '430', '434', '435', '438', '439', '440', '445', '446', '447', '451', '452', '459',
                          '462', '467', '473', '479', '480', '481', '482', '483', '489', '490', '492', '495', '497',
                          '500']

        # NOTE: Hard_Set_Test
        self.hard_list = ['18', '182', '186', '190', '526', '538', '663', '666', '667', '669', '670', '671', '672',
                          '675', '677', '678', '679', '680', '683', '684', '692', '693', '697', '698', '701', '703',
                          '706', '709', '711', '712', '713', '714', '716', '717', '719', '720', '725', '726', '728',
                          '730', '733', '736', '739', '740', '741', '744', '747', '748', '749', '750', '754', '759',
                          '761', '762', '763', '764', '768', '769', '770', '771', '774', '776', '777', '782', '783',
                          '789', '791', '793', '795', '796', '798', '800', '801', '804', '805', '806', '807', '808',
                          '809', '810', '812', '815', '818', '820', '821', '822', '826', '827', '829', '830', '833',
                          '834', '836', '837', '839', '840', '844', '846', '847', '848', '850', '851', '852', '854',
                          '856', '859', '860', '862', '864', '865', '866', '867', '868', '872', '873', '874', '875',
                          '881', '882', '883', '884', '885', '886', '887', '890', '891', '893', '894', '897', '898',
                          '899', '902', '904', '907', '908', '909', '910', '912', '914', '916', '923', '924', '925',
                          '927', '929', '938', '941', '942', '943', '944', '945', '952', '953', '956', '957', '960',
                          '962', '963', '964', '965', '969', '971', '972', '973', '974', '975', '976', '977', '978',
                          '981', '982', '985', '986', '994', '995', '997', '1000']

        # NOTE: Joint_Set_Test
        self.joint_list = ['2', '6', '7', '8', '12', '19', '20', '21', '22', '26', '28', '29', '31', '32', '33',
                           '38', '40', '41', '43', '44', '45', '47', '50', '51', '55', '57', '58', '59', '60', '61',
                           '64', '67', '69', '70', '76', '80', '81', '84', '90', '92', '96', '98', '99', '100',
                           '101', '103', '107', '108', '109', '112', '114', '115', '116', '118', '119', '120', '122',
                           '124', '127', '128', '129', '131', '132', '135', '137', '139', '140', '141', '142', '143',
                           '145', '147', '150', '151', '153', '157', '159', '160', '162', '163', '166', '167', '174',
                           '176', '179', '188', '191', '192', '195', '197', '198', '202', '203', '204', '206', '209',
                           '210', '211', '214', '220', '222', '223', '226', '237', '243', '244', '245', '247', '249',
                           '251', '252', '254', '255', '256', '258', '260', '262', '263', '264', '266', '275', '277',
                           '278', '279', '281', '282', '284', '285', '287', '288', '290', '295', '296', '306', '307',
                           '320', '322', '323', '326', '332', '335', '336', '337', '338', '340', '341', '343', '347',
                           '348', '349', '350', '352', '356', '361', '367', '370', '373', '374', '376', '381', '382',
                           '384', '385', '386', '391', '394', '396', '397', '398', '400', '403', '405', '406', '408',
                           '409', '410', '411', '412', '414', '415', '416', '417', '419', '420', '423', '424', '426',
                           '427', '430', '434', '435', '438', '439', '440', '445', '446', '447', '451', '452', '459',
                           '462', '467', '473', '479', '480', '481', '482', '483', '489', '490', '492', '495', '497',
                           '500', '18', '182', '186', '190', '526', '538', '663', '666', '667', '669', '670', '671',
                           '672', '675', '677', '678', '679', '680', '683', '684', '692', '693', '697', '698', '701',
                           '703', '706', '709', '711', '712', '713', '714', '716', '717', '719', '720', '725', '726',
                           '728', '730', '733', '736', '739', '740', '741', '744', '747', '748', '749', '750', '754',
                           '759', '761', '762', '763', '764', '768', '769', '770', '771', '774', '776', '777', '782',
                           '783', '789', '791', '793', '795', '796', '798', '800', '801', '804', '805', '806', '807',
                           '808', '809', '810', '812', '815', '818', '820', '821', '822', '826', '827', '829', '830',
                           '833', '834', '836', '837', '839', '840', '844', '846', '847', '848', '850', '851', '852',
                           '854', '856', '859', '860', '862', '864', '865', '866', '867', '868', '872', '873', '874',
                           '875', '881', '882', '883', '884', '885', '886', '887', '890', '891', '893', '894', '897',
                           '898', '899', '902', '904', '907', '908', '909', '910', '912', '914', '916', '923', '924',
                           '925', '927', '929', '938', '941', '942', '943', '944', '945', '952', '953', '956', '957',
                           '960', '962', '963', '964', '965', '969', '971', '972', '973', '974', '975', '976', '977',
                           '978', '981', '982', '985', '986', '994', '995', '997', '1000']

        self.easy_set = set(self.easy_list)
        self.hard_set = set(self.hard_list)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.easy_list, int(len(self.easy_list) * data_fraction))
        else:
            self.sequence_list = self.easy_list

        # self.sequence_list = ['2']  # for local test
        print(f'Dataset: COMTB_test, seq length: {len(self.sequence_list)}')

    def get_name(self):
        return 'COMTB_testingSet'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth_rect.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_modality_anno(self, seq_path):
        # 兼容 easy 和 hard 数据集
        file_paths = [
            os.path.join(seq_path, 'modality.tag'),
            os.path.join(seq_path, 'rgb-nir.txt')
        ]

        modality_anno_file = None
        for file_path in file_paths:
            if os.path.isfile(file_path):
                modality_anno_file = file_path
                break

        if modality_anno_file is None:
            raise FileNotFoundError("Neither 'rgb-nir.txt' nor 'modality.tag' found in the provided sequence path.")

        gt = np.loadtxt(modality_anno_file)
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        modality = self._read_modality_anno(seq_path)  # B, 1

        # 增强数据校验
        valid1 = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid2 = (modality[:] == 0) | (modality[:] == 1) | (modality[:] == 2) | (modality[:] == 3)
        try:
            valid = valid1 & valid2[:len(valid1)]
        except:
            print('Error seq_path:', seq_path)
            valid = valid1

        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'modality': modality}

    def _get_frame(self, seq_path, frame_id):
        frame_path = os.path.join(seq_path, 'img', sorted([p for p in os.listdir(os.path.join(seq_path, 'img')) if
                                                           os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])[
            frame_id])
        return self.image_loader(frame_path)

    def _get_modality_label(self, seq_path, frame_id):
        # 兼容 easy 和 hard 数据集
        file_paths = [
            os.path.join(seq_path, 'modality.tag'),
            os.path.join(seq_path, 'rgb-nir.txt')
        ]

        modality_anno_file = None
        for file_path in file_paths:
            if os.path.isfile(file_path):
                modality_anno_file = file_path
                break

        rgb_nir_label = np.loadtxt(modality_anno_file)[frame_id]
        return rgb_nir_label

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]
        rgb_nir_label = [self._get_modality_label(seq_path, f) for f in frame_ids]

        # print('get_frames frame_list', len(frame_list))
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        if seq_id in self.easy_set:
            # Easy 数据集, 当前0-RGB,1-NIR,2用1代替
            rgb_nir_label = [1 if i == 2 else i for i in rgb_nir_label]
        else:
            # Hard 数据集, 当前0-RGB,1用0代替,2-NIR,3用2代替
            rgb_nir_label = [0 if i == 1 else 2 if i == 3 else i for i in rgb_nir_label]
            rgb_nir_label = [1 if i == 2 else i for i in rgb_nir_label]  # 为了统一NIR用1表示

        modality_label = torch.from_numpy(np.array(rgb_nir_label)).float()
        return frame_list, anno_frames, object_meta, modality_label
