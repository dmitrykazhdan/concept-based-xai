import numpy as np

from datasets.latentFactorData import LatentFactorData, get_task_data, built_task_fn

'''
See https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
For a nice overview 

6 latent factors:   color, shape, scale, rotation and position (x and y)
'latents_sizes':    array([ 1,  3,  6, 40, 32, 32])
'''
DSPRITES_concept_names  = ['shape', 'scale', 'rotation', 'x_pos', 'y_pos']
DSPRITES_concept_n_vals = [3, 6, 40, 32, 32]

class dSprites(LatentFactorData):

    def __init__(self, dataset_path, task_name='shape_scale_small_skip', train_size=0.85, random_state=42):
        '''
        :param dataset_path:  path to the .npz dsprites file
        :param task_name: the task to use with the dataset for creating labels
        '''

        # Note: we exclude the trivial 'color' concept
        super().__init__(dataset_path=dataset_path, task_name=task_name, num_factors=5,
                         sample_shape=[64, 64, 1], c_names=[DSPRITES_concept_names],
                         task_fn=DSPRITES_TASKS[task_name])
        self._get_generators(train_size, random_state)

    def _load_x_c_data(self):
        # Load dataset
        dataset_zip = np.load(self.dataset_path)
        x_data = dataset_zip['imgs']
        x_data = np.expand_dims(x_data, axis=-1)
        c_data = dataset_zip['latents_classes']
        c_data = c_data[:, 1:]  # Remove color concept
        return x_data, c_data


# ===========================================================================
#                   Task DEFINITIONS
# ===========================================================================

def get_small_skip_ranges_filter_fn():
    '''
    Filter out certain values only
    '''
    ranges = [list(range(3)), list(range(6)), list(range(0, 40, 5)),
              list(range(0, 32, 2)), list(range(0, 32, 2))]

    def filter_fn(concept):
        return all([(concept[i] in ranges[i]) for i in range(len(ranges))])

    return filter_fn


def shape_label_fn(c_data):
    return c_data[0]


def shape_scale_label_fn(n_shapes, n_scales):
    label_map = {}
    cnt = 0

    for sh in range(n_shapes):
        for sc in range(n_scales):
            key = sh * n_scales + sc
            label_map[key] = cnt
            cnt += 1

    def label_fn(c_data):
        key = c_data[0] * n_scales + c_data[1]
        return label_map[key]

    return label_fn


def get_shape_full(x_data, c_data):
    label_fn = shape_label_fn
    return get_task_data(x_data, c_data, label_fn, filter_fn=None)


def get_shape_small_skip(x_data, c_data):
    filter_fn = get_small_skip_ranges_filter_fn()
    label_fn = shape_label_fn
    return get_task_data(x_data, c_data, label_fn, filter_fn)


def get_shape_scale_full(x_data, c_data):
    label_fn = shape_scale_label_fn(3, 6)
    return get_task_data(x_data, c_data, label_fn, filter_fn=None)


def get_shape_scale_small_skip(x_data, c_data):
    filter_fn = get_small_skip_ranges_filter_fn()
    label_fn = shape_scale_label_fn(3, 6)
    return get_task_data(x_data, c_data, label_fn, filter_fn)



# ===========================================================================
#                   Define task function lookups
# ===========================================================================

DSPRITES_TASKS = {"shape_full":             get_shape_full,
                  "shape_small_skip":       get_shape_small_skip,
                  "shape_scale_full":       get_shape_scale_full,
                  "shape_scale_small_skip": get_shape_scale_small_skip,
                  }
