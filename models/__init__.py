import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'video_base3':
        from .Video_base_model3 import VideoBaseModel as M
    elif model == 'video_base4':
        from .Video_base_model4 import VideoBaseModel as M
    elif model == 'video_base4_m':
        from .Video_base_model4_m import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
