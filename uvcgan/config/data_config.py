import logging

# from uvcgan.consts      import MERGE_PAIRED, MERGE_UNPAIRED, MERGE_NONE
# from uvcgan.utils.funcs import check_value_in_range

from .config_base import ConfigBase

LOGGER      = logging.getLogger('uvcgan.config')

class DatasetConfig(ConfigBase):
    """Dataset configuration.

    Parameters
    ----------
    dataset : str or dict
        Dataset specification.
    shape : tuple of int
        Shape of inputs.
    transform_train : None or str or dict or list of those
        Transformations to be applied to the training dataset.
        If `transform_train` is None, then no transformations will be applied
        to the training dataset.
        If `transform_train` is str, then its value is interpreted as a name
        of the transformation.
        If `transform_train` is dict, then it is expected to be of the form
        `{ 'name' : TRANFORM_NAME, **kwargs }`, where 'name' is the name of
        the transformation, and `kwargs` dict will be passed to the
        transformation constructor.
        Otherwise, `transform_train` is expected to be a list of values above.
        The corresponding transformations will be chained together in the
        order that they are specified.
        Default: None.
    transform_val : None or str or dict or list of those
        Transformations to be applied to the validation dataset.
        C.f. `transform_train`.
        Default: None.
    """

    __slots__ = [
        'dataset',
        'shape',
        'transform_train',
        'transform_val',
    ]

    def __init__(
        self, dataset, shape,
        transform_train = None,
        transform_val = None,
    ):
        super().__init__()

        self.dataset         = dataset
        self.shape           = shape
        self.transform_train = transform_train
        self.transform_val  = transform_val

class DataConfig(ConfigBase):
    """Data configuration.

    Parameters
    ----------
    datasets : list of dict
        List of dataset specifications.
    merge_type : str, optional
        How to merge samples from datasets.
        Choices: 'paired', 'unpaired', 'none'.
        Default: 'unpaired'
    workers : int, optional
        Number of data workers.
        Default: None
    """

    __slots__ = [
        'datasets',
    ]

    def __init__(self, datasets):
        super().__init__()

        # check_value_in_range(merge_type, MERGE_TYPES, 'merge_type')
        assert isinstance(datasets, list)

        self.datasets    = [ DatasetConfig(**x) for x in datasets ]
        # self.merge_type  = merge_type
        # self.workers     = workers

def parse_data_config(data, data_args, image_shape):
    return DataConfig(**data)
