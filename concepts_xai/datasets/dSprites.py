'''
See https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
for a nice overview.

6 latent factors: color, shape, scale, rotation and position (x and y)
'latents_sizes': [1,  3,  6, 40, 32, 32]
'''

import numpy as np

from .latentFactorData import LatentFactorData, get_task_data

################################################################################
## GLOBAL VARIABLES
################################################################################

CONCEPT_NAMES = [
    'shape',
    'scale',
    'rotation',
    'x_pos',
    'y_pos',
]

CONCEPT_N_VALUES = [
    3,  # [square, ellipse, heart]
    6,  # np.linspace(0.5, 1, 6)
    40,  # 40 values in {0, 1, ..., 39} representing angles in [0, 2 * pi]
    32,  # 32 values in {0, 2, 3, ..., 31} representing coordinates in [0, 1]
    32,  # 32 values in {0, 2, 3, ..., 31} representing coordinates in [0, 1]
]


################################################################################
## DATASET LOADER
################################################################################

class dSprites(LatentFactorData):

    def __init__(
        self,
        dataset_path,
        task='shape_scale_small_skip',
        train_size=0.85,
        random_state=None,
    ):
        '''
        :param str dataset_path: path to the .npz dsprites file.
        :param Or[
            str,
            Function[(ndarray, ndarray), (ndarray, ndarray, ndarray)
        ] task: the task to use with the dataset for creating
            labels. If this is a string, then it must be the name of a
            pre-defined task in the DSPRITES_TASKS lookup table. Otherwise
            we expect a function that takes two np.ndarrays (x_data, c_data),
            corresponding to the dSprites samples and their respective concepts
            respectively, and produces a tuple of three np.ndarrays
            (x_data, c_data, y_data) corresponding to the task's
            samples, ground truth concept values, and labels, respectively.
        '''

        # Note: we exclude the trivial 'color' concept
        if isinstance(task, str):
            if task not in DSPRITES_TASKS:
                raise ValueError(
                    f'If the given task is a string, then it is expected to be '
                    f'the name of a pre-defined task in '
                    f'{list(DSPRITES_TASKS.keys())}. However, we were given '
                    f'"{task}" which is not a known task.'
                )
            task_fn = DSPRITES_TASKS[task]
        else:
            task_fn = task
        super().__init__(
            dataset_path=dataset_path,
            task_name="dSprites",
            num_factors=len(CONCEPT_NAMES),
            sample_shape=[64, 64, 1],
            c_names=CONCEPT_NAMES,
            task_fn=task_fn,
        )
        self._get_generators(train_size, random_state)

    def _load_x_c_data(self):
        # Load dataset
        dataset_zip = np.load(self.dataset_path)
        x_data = dataset_zip['imgs']
        x_data = np.expand_dims(x_data, axis=-1)
        c_data = dataset_zip['latents_classes']
        c_data = c_data[:, 1:]  # Remove color concept
        return x_data, c_data


################################################################################
# TASK DEFINITIONS
################################################################################

def cardinality_encoding(card_group_1, card_group_2):
    result_to_encoding = {}
    for i in card_group_1:
        for j in card_group_2:
            result_to_encoding[(i, j)] = len(result_to_encoding)
    return result_to_encoding


def small_skip_ranges_filter_fn(concept):
    '''
    Filter out certain values only
    '''
    ranges = [
        list(range(3)),
        list(range(6)),
        list(range(0, 40, 5)),
        list(range(0, 32, 2)),
        list(range(0, 32, 2)),
    ]
    return all([
        (concept[i] in ranges[i]) for i in range(len(ranges))
    ])


def get_shape_full(x_data, c_data):
    return get_task_data(
        x_data=x_data,
        c_data=c_data,
        label_fn=lambda c_data: c_data[0],
    )


def get_shape_small_skip(x_data, c_data):
    return get_task_data(
        x_data=x_data,
        c_data=c_data,
        label_fn=lambda c_data: c_data[0],
        filter_fn=small_skip_ranges_filter_fn,
    )


def get_shape_scale_full(x_data, c_data):
    return get_task_data(
        x_data=x_data,
        c_data=c_data,
        label_fn=lambda c_data: (
            c_data[0] * CONCEPT_N_VALUES[1] + c_data[1]
        ),
    )


def get_shape_scale_small_skip(x_data, c_data):
    label_remap = cardinality_encoding(
        list(range(3)),
        list(range(6)),
    )
    return get_task_data(
        x_data=x_data,
        c_data=c_data,
        label_fn=lambda c_data: label_remap[(c_data[0], c_data[1])],
        filter_fn=small_skip_ranges_filter_fn,
    )


################################################################################
# TASK FUNCTION LOOKUP TABLE
################################################################################

DSPRITES_TASKS = {
    "shape_full": get_shape_full,
    "shape_scale_full": get_shape_scale_full,
    "shape_scale_small_skip": get_shape_scale_small_skip,
    "shape_small_skip": get_shape_small_skip,
}
