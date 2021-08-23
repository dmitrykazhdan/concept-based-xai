import itertools
import numpy as np
import h5py

from .latentFactorData import LatentFactorData, get_task_data

################################################################################
## GLOBAL VARIABLES
################################################################################


CONCEPT_NAMES = [
    'floor_hue',
    'wall_hue',
    'object_hue',
    'scale',
    'shape',
    'orientation',
]
CONCEPT_N_VALUES = [
    10,
    10,
    10,
    8,
    4,
    15,
]


################################################################################
## DATASET LOADER
################################################################################

class shapes3D(LatentFactorData):

    def __init__(
        self,
        dataset_path,
        task='shape_full',
        train_size=0.85,
        random_state=None,
    ):
        '''
        :param dataset_path:  path to the .npz shapes3D file
        :param Or[
            str,
            Function[(ndarray, ndarray), (ndarray, ndarray, ndarray)
        ] task: the task to use with the dataset for creating
            labels. If this is a string, then it must be the name of a
            pre-defined task in the SHAPES3D_TASKS lookup table. Otherwise
            we expect a function that takes two np.ndarrays (x_data, c_data),
            corresponding to the dSprites samples and their respective concepts
            respectively, and produces a tuple of three np.ndarrays
            (x_data, c_data, y_data) corresponding to the task's
            samples, ground truth concept values, and labels, respectively.
        '''

        if isinstance(task, str):
            if task not in SHAPES3D_TASKS:
                raise ValueError(
                    f'If the given task is a string, then it is expected to be '
                    f'the name of a pre-defined task in '
                    f'{list(SHAPES3D_TASKS.keys())}. However, we were given '
                    f'"{task}" which is not a known task.'
                )
            task_fn = SHAPES3D_TASKS[task]
        else:
            task_fn = task

        super().__init__(
            dataset_path=dataset_path,
            task_name="3dshapes",
            num_factors=len(CONCEPT_NAMES),
            sample_shape=[64, 64, 3],
            c_names=CONCEPT_NAMES,
            task_fn=task_fn,
        )
        self._get_generators(train_size, random_state)

    def _load_x_c_data(self):
        # Get concept data
        latent_size_lists = [list(np.arange(i)) for i in CONCEPT_N_VALUES]
        c_data = np.array(list(itertools.product(*latent_size_lists)))
        # Load image data
        with h5py.File(self.dataset_path, 'r') as hf:
            x_data = np.array(hf.get('images')) / 255.


        return x_data, c_data


################################################################################
# TASK DEFINITIONS
################################################################################

def small_skip_ranges_filter_fn(concept):
    '''
    Filter out certain values only
    '''
    ranges = [
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 8, 2)),
        list(range(4)),
        list(range(0, 15, 2)),
    ]

    return all([(concept[i] in ranges[i]) for i in range(len(ranges))])


def get_shape_full(x_data, c_data):
    return get_task_data(
        x_data=x_data,
        c_data=c_data,
        label_fn=lambda x: x[4],
    )


def get_shape_small_skip(x_data, c_data):
    return get_task_data(
        x_data=x_data,
        c_data=c_data,
        label_fn=lambda x: x[4],
        filter_fn=small_skip_ranges_filter_fn,
    )


def get_reduced_filter_fn():
    ranges = [
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 8, 2)),
        list(range(4)),
        list(range(15)),
    ]

    def filter_fn(concept):
        return all([(concept[i] in ranges[i]) for i in range(len(ranges))])

    return filter_fn


def get_reduced_shapes3d(x_data, c_data):

    ranges = [
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 8, 2)),
        list(range(4)),
        list(range(15)),
    ]

    label_fn = shape_label_fn

    def filter_fn(concept):
        return all([(concept[i] in ranges[i]) for i in range(len(ranges))])

    return get_task_data(
        x_data=x_data,
        c_data=c_data,
        label_fn=shape_label_fn,
        filter_fn=filter_fn,
    )


################################################################################
# TASK FUNCTION LOOKUP TABLE
################################################################################

SHAPES3D_TASKS = {
    "reduced_shapes3d" : get_reduced_shapes3d,
    "shape_full" : get_shape_full,
    "shape_small_skip" : get_shape_small_skip,
}


