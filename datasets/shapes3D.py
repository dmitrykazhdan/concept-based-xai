import itertools
import numpy as np
import h5py

from datasets.latentFactorData import LatentFactorData, get_task_data, built_task_fn

SHAPES3D_concept_names  = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
SHAPES3D_concept_n_vals = [10, 10, 10, 8, 4, 15]

class shapes3D(LatentFactorData):

    def __init__(self, dataset_path, task_name='shape_full', train_size=0.85, random_state=42):
        '''
        :param dataset_path:  path to the .npz shapes3D file
        :param task_name: the task to use with the dataset for creating labels
        '''
        super().__init__(dataset_path=dataset_path, task_name=task_name, num_factors=6,
                         sample_shape=[64, 64, 3],
                         c_names=SHAPES3D_concept_names,
                         task_fn=SHAPES3D_TASKS[task_name])
        self._get_generators(train_size, random_state)


    def _load_x_c_data(self):

        # Get concept data
        latent_sizes = np.array([10, 10, 10, 8, 4, 15])
        latent_size_listss = [list(np.arange(i)) for i in latent_sizes]
        c_data = np.array(list(itertools.product(*latent_size_listss)))
        # Load image data
        hf = h5py.File(self.dataset_path, 'r')
        x_data = np.array(hf.get('images')) / 255.

        return x_data, c_data

# ===========================================================================
#                   Task DEFINITIONS
# ===========================================================================

def get_small_skip_ranges_filter_fn():
    '''
    Filter out certain values only
    '''
    ranges      = [list(range(0, 10, 2)), list(range(0, 10, 2)), list(range(0, 10, 2)),
                   list(range(0, 8, 2)), list(range(4)), list(range(0, 15, 2))]

    def filter_fn(concept):
        return all([(concept[i] in ranges[i]) for i in range(len(ranges))])

    return filter_fn


def shape_label_fn(c_data):
    return c_data[4]


def get_shape_full(x_data, c_data):
    label_fn  = shape_label_fn
    return get_task_data(x_data, c_data, label_fn, filter_fn=None)


def get_shape_small_skip(x_data, c_data):
    filter_fn = get_small_skip_ranges_filter_fn()
    label_fn  = shape_label_fn
    return get_task_data(x_data, c_data, label_fn, filter_fn)


def get_reduced_filter_fn():
    ranges      = [list(range(0, 10, 2)), list(range(0, 10, 2)), list(range(0, 10, 2)),
                   list(range(0, 8, 2)), list(range(4)), list(range(15))]
    def filter_fn(concept):
        return all([(concept[i] in ranges[i]) for i in range(len(ranges))])

    return filter_fn


def get_reduced_shapes3d(x_data, c_data):

    ranges      = [list(range(0, 10, 2)), list(range(0, 10, 2)), list(range(0, 10, 2)),
                   list(range(0, 8, 2)), list(range(4)), list(range(15))]

    label_fn  = shape_label_fn

    def filter_fn(concept):
        return all([(concept[i] in ranges[i]) for i in range(len(ranges))])

    return get_task_data(x_data, c_data, label_fn, filter_fn)


# ===========================================================================
#                   Define task function lookups
# ===========================================================================

SHAPES3D_TASKS = {"shape_full"              : get_shape_full,
                  "shape_small_skip"        : get_shape_small_skip,
                  "reduced_shapes3d"        : get_reduced_shapes3d,
}


