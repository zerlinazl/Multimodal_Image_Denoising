import argparse
import os

from uvcgan import ROOT_OUTDIR, train
from uvcgan.utils.parsers import add_preset_name_parser, add_batch_size_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Train ImageNet BERT')
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'vit-unet-12')
    add_batch_size_parser(parser, default = 4)
    return parser.parse_args()

GEN_PRESETS = {
    'vit-unet-6' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 6,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [48, 96, 192, 384],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : 'instance',
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : 'sigmoid',
        },
    },
    'vit-unet-12' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 12,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [48, 96, 192, 384],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : 'instance',
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : 'sigmoid',
        },
    },
    # change model params here!!
    'vit-unet-12-multimodal' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 12,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [48, 96, 192, 383],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : 'instance',
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : 'sigmoid',
            'meta'               : True,
        },
    },
}

# 'datasets' : [
#                 {
#                     'dataset' : {
#                         'name'   : 'ndarray-domain-hierarchy',
#                         'domain' : domain,
#                         'path'   : 'slats_tiles_excerpt',
#                     },
#                     'shape'           : (1, 256, 256),
#                     'transform_train' : None,
#                     'transform_test'  : None,
#                 } for domain in [ 'fake', 'real' ]
#             ],
#             'merge_type' : 'unpaired'

cmdargs = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
                {
                    'dataset' : {
                        'name'   : 'sidd',
                        'domain' : domain,
                        'path'   : '/home/zlai/data/SIDD_S_sRGB/',
                    },
                    'shape'           : (3, 256, 256),
                    'transform_train' : [
                        {
                            'name' : 'center-crop',
                            'size' : 256,
                            # 'pad_if_needed' : True
                        },
                    ],
                    'transform_val' : [
                        {
                            'name' : 'center-crop',
                            'size' : 256,
                        },
                    ],
                } for domain in [ 'N', 'GT' ]
            ],
    },
    'image_shape' : (3, 256, 256),
    'epochs'      : 120,
    'discriminator' : None,
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : cmdargs.batch_size * 5e-3 / 512,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        }
    },
    'model'      : 'simple-autoencoder', # uses uvcgan/cgan/simple_autoencoder.py
    'model_args' : {
    },
    'scheduler' : {
        'name'      : 'CosineAnnealingWarmRestarts',
        'T_0'       : 100,
        'T_mult'    : 1,
        'eta_min'   : cmdargs.batch_size * 5e-8 / 512,
    },
    'loss'             : 'l2',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
# args
    'label'      : f'{cmdargs.gen}-256',
    'outdir'     : os.path.join(ROOT_OUTDIR, 'SIDD'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}

train(args_dict)

