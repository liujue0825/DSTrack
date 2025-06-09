from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    # NOTE: Cross-modality
    settings.comtb_path = '/DATA/RGBNIR'

    settings.prj_dir = '/DATA/liujue/DSTrack'
    settings.result_plot_path = '/DATA/liujue/DSTrack/output/test/result_plots'
    settings.results_path = '/DATA/liujue/DSTrack/output/test/tracking_results'  # Where to store tracking results
    settings.save_dir = '/DATA/liujue/DSTrack/output'
    settings.network_path = '/DATA/liujue/DSTrack/output/test/networks'  # Where tracking networks are stored.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/DATA/liujue/OSTrack/data/got10k_lmdb'
    settings.got10k_path = '/DATA/liujue/OSTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/DATA/liujue/OSTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/DATA/liujue/OSTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/DATA/liujue/OSTrack/data/lasot_lmdb'
    settings.lasot_path = '/DATA/liujue/OSTrack/data/lasot'
    settings.nfs_path = '/DATA/liujue/OSTrack/data/nfs'
    settings.otb_path = '/DATA/liujue/OSTrack/data/otb'
    settings.segmentation_path = '/DATA/liujue/OSTrack/output/test/segmentation_results'
    settings.tc128_path = '/DATA/liujue/OSTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/DATA/liujue/OSTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/DATA/liujue/OSTrack/data/trackingnet'
    settings.uav_path = '/DATA/liujue/OSTrack/data/uav'
    settings.vot18_path = '/DATA/liujue/OSTrack/data/vot2018'
    settings.vot22_path = '/DATA/liujue/OSTrack/data/vot2022'
    settings.vot_path = '/DATA/liujue/OSTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings
