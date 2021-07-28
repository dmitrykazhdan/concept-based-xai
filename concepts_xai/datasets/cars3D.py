import os

import PIL
import numpy as np
import scipy.io as sio
import tensorflow as tf

from .latentFactorData import LatentFactorData, get_task_data, built_task_fn

CARS_concept_names  = ['elevation', 'azimuth', 'object_type']
CARS_concept_n_vals = [4, 24, 183]

class Cars3D(LatentFactorData):

    def __init__(self, dataset_path, task_name='elevation_full', train_size=0.85, random_state=42):
        '''
        :param dataset_path:  path to the cars dataset folder
        :param task_name: the task to use with the dataset for creating labels
        '''
        super().__init__(dataset_path=dataset_path, task_name=task_name, num_factors=3,
                         sample_shape=[64, 64, 3], c_names=CARS_concept_names,
                         task_fn=CARS3D_TASKS[task_name])
        self._get_generators(train_size, random_state)

    def _load_x_c_data(self):
        x_data = []
        all_files = [x for x in tf.io.gfile.listdir(self.dataset_path) if ".mat" in x]
        c_data = []

        for i, filename in enumerate(all_files):
            data_mesh = load_mesh(os.path.join(self.dataset_path, filename))
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i, len(factor1) * len(factor2))
            ])

            c_data += [list(all_factors[j]) for j in range(all_factors.shape[0])]
            x_data.append(data_mesh)

        x_data = np.concatenate(x_data)
        c_data = np.array(c_data)

        return x_data, c_data


def load_mesh(filename):
    """Parses a single source file and rescales contained images."""
    with tf.io.gfile.GFile(filename, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        flattened_im = flattened_mesh[i, :, :, :]
        pic = PIL.Image.fromarray(flattened_im)
        pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
        np_pic = np.array(pic)
        rescaled_mesh[i, :, :, :] = np_pic
    return rescaled_mesh * 1. / 255


# ===========================================================================
#                   Task DEFINITIONS
# ===========================================================================

def get_elevation_full(x_data, c_data):
    label_fn = lambda c: c[0]
    return get_task_data(x_data, c_data, label_fn, filter_fn=None)


def get_all_concepts(x_data, c_data):
    label_fn = lambda c: c
    return get_task_data(x_data, c_data, label_fn, filter_fn=None)


# ===========================================================================
#                   Define task function lookups
# ===========================================================================

CARS3D_TASKS = {
        'all_concepts':                         get_all_concepts,
        'elevation_full':                       get_elevation_full,
        "bin_elevation":                        built_task_fn(lambda c: int(c[0] >= 2)),
}
