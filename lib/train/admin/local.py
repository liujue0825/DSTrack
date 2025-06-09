class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/DATA/liujue/DSTrack'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/DATA/liujue/DSTrack/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = '/DATA/liujue/DSTrack/pretrained_networks'

        # NOTE: Cross-modality
        self.comtb_trainingset_dir = '/DATA/RGBNIR'
        self.comtb_testingset_dir = '/DATA/RGBNIR'

        self.lasot_dir = '\\DATA\\liujue\\OSTrack\\data\\lasot'
        self.got10k_dir = '\\DATA\\liujue\\OSTrack\\data\\got10k\\train'
        self.got10k_val_dir = '\\DATA\\liujue\\OSTrack\\data\\got10k\\val'
        self.lasot_lmdb_dir = '\\DATA\\liujue\\OSTrack\\data\\lasot_lmdb'
        self.got10k_lmdb_dir = '\\DATA\\liujue\\OSTrack\\data\\got10k_lmdb'
        self.trackingnet_dir = '\\DATA\\liujue\\OSTrack\\data\\trackingnet'
        self.trackingnet_lmdb_dir = '\\DATA\\liujue\\OSTrack\\data\\trackingnet_lmdb'
        self.coco_dir = '\\DATA\\liujue\\OSTrack\\data\\coco'
        self.coco_lmdb_dir = '\\DATA\\liujue\\OSTrack\\data\\coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '\\DATA\\liujue\\OSTrack\\data\\vid'
        self.imagenet_lmdb_dir = '\\DATA\\liujue\\OSTrack\\data\\vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
