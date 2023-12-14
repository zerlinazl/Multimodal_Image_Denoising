import os
import torch

import torchvision

from uvcgan.consts         import ROOT_DATA
from .datasets.celeba      import CelebaDataset
from .datasets.cyclegan    import CycleGANDataset
from .datasets.cyclegan_v2 import CycleGANv2Dataset
from .transforms           import select_transform
from .utils                import imbalanced_collate
from uvcgan.torch.select   import extract_name_kwargs
from .datasets.custom_dataset           import custom_dataset
from .datasets.zipper      import DatasetZipper

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'

def worker_init_fn(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset

    try:
        dataset.reseed(worker_seed)
    except AttributeError:
        pass

# def load_sidd_datasets(transform_train, transform_val, path, **data_args):
    # dset_train = torchvision.datasets.ImageFolder(
    #     os.path.join(path, 'train'), transform = transform_train, **data_args
    # )
    # dset_val = torchvision.datasets.ImageFolder(
    #     os.path.join(path, 'val'), transform = transform_val, **data_args
    # )
    # dset_test = torchvision.datasets.ImageFolder(
    #     os.path.join(path, 'test'), transform = transform_val, **data_args
    # )

    # return (dset_train, dset_val)


#11/20
def construct_data_loaders(data_config, batch_size, split):
    datasets = construct_datasets(data_config, split)
    shuffle  = (split == SPLIT_TRAIN)

    # if data_config.merge_type == MERGE_PAIRED:
    dataset = DatasetZipper(datasets)

    return construct_single_loader(
        dataset, batch_size, shuffle, None,
        drop_last = False
    )
    
def construct_single_loader(dataset, batch_size, shuffle,
    workers         = None,
    prefetch_factor = 20,
    **kwargs
):
    if workers is None:
        workers = min(torch.get_num_threads(), 20)

    return torch.utils.data.DataLoader(
        dataset, batch_size,
        shuffle         = shuffle,
        num_workers     = workers,
        prefetch_factor = prefetch_factor,
        pin_memory      = True,
        **kwargs
    )

def construct_single_dataset(dataset_config, split):
    name, kwargs = extract_name_kwargs(dataset_config.dataset)
    path         = os.path.join(ROOT_DATA, kwargs.pop('path', name))

    if split == SPLIT_TRAIN:
        transform = select_transform(dataset_config.transform_train)
    else:
        transform = select_transform(dataset_config.transform_val)

    return select_dataset(name, path, split, transform, dataset_config.dataset, **kwargs)

def construct_datasets(data_config, split):
    for config in data_config.datasets:
        return [
            construct_single_dataset(config, split)
                for config in data_config.datasets
        ]

def select_dataset(name, path, split, transform, dataset, **kwargs):
    if name == 'custom' or name == 'sidd':
        # assert 'dataset' in kwargs, \
        #     'a path to your dataset API must provided'
        # dataset = kwargs.pop('dataset')
        return custom_dataset(dataset, path, split = split, **kwargs)


# def load_cyclegan_datasets(transform_train, transform_val, path, **data_args):
#     dset_train = CycleGANDataset(
#         path, transform = transform_train, is_train = True, **data_args
#     )
#     dset_val = CycleGANDataset(
#         path, transform = transform_val, is_train = False, **data_args
#     )

#     return (dset_train, dset_val)

# def load_cyclegan_v2_datasets(
#     transform_train, transform_val, path, **data_args
# ):
#     dset_train = CycleGANv2Dataset(
#         path, transform = transform_train, is_train = True, **data_args
#     )
#     dset_val = CycleGANv2Dataset(
#         path, transform = transform_val, is_train = False, **data_args
#     )

#     return (dset_train, dset_val)

# def load_imagenet_datasets(transform_train, transform_val, path, **data_args):
#     dset_train = torchvision.datasets.ImageNet(
#         path, transform = transform_train, split = 'train', **data_args
#     )
#     dset_val = torchvision.datasets.ImageNet(
#         path, transform = transform_val, split = 'val', **data_args
#     )

#     return (dset_train, dset_val)

# def load_imagedir_datasets(transform_train, transform_val, path, **data_args):
#     dset_train = torchvision.datasets.ImageFolder(
#         os.path.join(path, 'train'), transform = transform_train, **data_args
#     )
#     dset_val = torchvision.datasets.ImageFolder(
#         os.path.join(path, 'val'), transform = transform_val, **data_args
#     )

#     return (dset_train, dset_val)

# def load_celeba_datasets(transform_train, transform_val, path, **data_args):
#     dset_train = CelebaDataset(
#         path, transform = transform_train, split = 'train', **data_args
#     )

#     dset_val = CelebaDataset(
#         path, transform = transform_val, split = 'test', **data_args
#     )

#     return (dset_train, dset_val)

# def select_datasets(
#     dataset, transform_train, transform_val, path = None, **dataset_args
# ):
#     path = os.path.join(ROOT_DATA, path or dataset)

#     if dataset == 'celeba':
#         return load_celeba_datasets(
#             transform_train, transform_val, path, **dataset_args
#         )

#     if dataset == 'cyclegan':
#         return load_cyclegan_datasets(
#             transform_train, transform_val, path, **dataset_args
#         )

#     if dataset == 'cyclegan-v2':
#         return load_cyclegan_v2_datasets(
#             transform_train, transform_val, path, **dataset_args
#         )

#     if dataset == 'imagenet':
#         return load_imagenet_datasets(
#             transform_train, transform_val, path, **dataset_args
#         )

#     if dataset in [ 'imagedir', 'image-folder' ]:
#         return load_imagedir_datasets(
#             transform_train, transform_val, path, **dataset_args
#         )
    
#     if dataset == "sidd":
#         path = "/home/zlai/data/SIDD_S_sRGB/"
#         return load_sidd_datasets(
#             transform_train, transform_val, path, **dataset_args
#         )

#     raise ValueError(f"Unknown dataset: '{dataset}'")

# def load_datasets(data_config):
#     transform_train = select_transform(data_config.transform_train)
#     transform_val   = select_transform(data_config.transform_val)

#     return select_datasets(
#         data_config.dataset, transform_train, transform_val,
#         **data_config.dataset_args
#     )

# def construct_loader(
#     dataset, batch_size, shuffle,
#     workers         = None,
#     prefetch_factor = 20,
#     **kwargs
# ):
#     if workers is None:
#         workers = min(torch.get_num_threads(), 20)

#     return torch.utils.data.DataLoader(
#         dataset, batch_size,
#         shuffle            = shuffle,
#         num_workers        = workers,
#         drop_last          = False,
#         prefetch_factor    = prefetch_factor,
#         collate_fn         = imbalanced_collate,
#         worker_init_fn     = worker_init_fn,
#         **kwargs
#     )

# def get_data(data_config, batch_size, workers):
#     datasets = load_datasets(data_config)

#     (it_train, it_val) = [
#         construct_loader(x, batch_size, shuffle = (i == 0), workers = workers)
#         for (i,x) in enumerate(datasets)
#     ]

#     return (it_train, it_val)

