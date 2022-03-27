"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'video_samesize_SDSD_test':
        from data.dataset_SDSD_test import VideoSameSizeDataset as D
    elif mode == 'video_samesize_SDSD_train':
        from data.dataset_SDSD_train import VideoSameSizeDataset as D
    elif mode == 'video_samesize_sid':
        from data.dataset_SID import VideoSameSizeDataset as D
    elif mode == 'video_samesize_lol':
        from data.dataset_LOLv1 import VideoSameSizeDataset as D
    elif mode == 'video_samesize_lol2':
        from data.dataset_LOLv2_real import VideoSameSizeDataset as D
    elif mode == 'video_samesize_lol3':
        from data.dataset_LOLv2_synthetic import VideoSameSizeDataset as D
    elif mode == 'video_samesize_SMID_test':
        from data.dataset_SMID_test import VideoSameSizeDataset as D
    elif mode == 'video_samesize_SMID_train':
        from data.dataset_SMID_train import VideoSameSizeDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
