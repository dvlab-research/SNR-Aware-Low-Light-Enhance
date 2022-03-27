import torch
import models.archs.low_light_transformer as low_light_transformer

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'low_light_transformer':
        netG = low_light_transformer.low_light_transformer(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                                           predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                                           w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

