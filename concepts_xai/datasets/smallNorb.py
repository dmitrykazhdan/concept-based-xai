import os

import PIL
import numpy as np
import tensorflow as tf

from .latentFactorData import LatentFactorData, get_task_data

SMALLNORB_concept_names  = ['category', 'instance', 'elevation', 'azimuth', 'lighting']
SMALLNORB_concept_n_vals = [5, 10, 9, 18, 6]

class SmallNorb(LatentFactorData):

    def __init__(self, dataset_path, task_name='category_full', train_size=0.85, random_state=42):
        '''
        :param dataset_path:  path to the smallnorb files directory
        :param task_name: the task to use with the dataset for creating labels
        '''

        super().__init__(dataset_path=dataset_path, task_name=task_name, num_factors=5,
                         sample_shape=[64, 64, 1], c_names=SMALLNORB_concept_names,
                         task_fn=SMALLNORB_TASKS[task_name])
        self._get_generators(train_size, random_state)

    def _load_x_c_data(self):
        files_dir = self.dataset_path
        filename_template = "smallnorb-{}-{}.mat"
        splits = ["5x46789x9x18x6x2x96x96-training", "5x01235x9x18x6x2x96x96-testing"]
        x_datas, c_datas = [], []

        for i, split in enumerate(splits):
            data_fname = os.path.join(files_dir, filename_template.format(splits[i], 'dat'))
            cat_fname = os.path.join(files_dir, filename_template.format(splits[i], 'cat'))
            info_fname = os.path.join(files_dir, filename_template.format(splits[i], 'info'))

            x_data = _read_binary_matrix(data_fname)
            x_data = _resize_images(x_data[:, 0])  # Resize data, and only retain data from 1 camera
            c_cat = _read_binary_matrix(cat_fname)
            c_info = _read_binary_matrix(info_fname)
            c_info = np.copy(c_info)
            c_info[:, 2:3] = c_info[:, 2:3] / 2  # Set azimuth values to be consecutive digits
            c_data = np.column_stack((c_cat, c_info))

            x_datas.append(x_data)
            c_datas.append(c_data)

        x_data = np.concatenate(x_datas)
        x_data = np.expand_dims(x_data, axis=-1)
        c_data = np.concatenate(c_datas)

        return x_data, c_data


def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = PIL.Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), PIL.Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images / 255.


def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    with tf.io.gfile.GFile(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double"
        }
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data


# ===========================================================================
#                   Task DEFINITIONS
# ===========================================================================

def get_category_full(x_data, c_data):
    label_fn = lambda c: c[0]
    return get_task_data(x_data, c_data, label_fn, filter_fn=None)


# ===========================================================================
#                   Define task function lookups
# ===========================================================================

SMALLNORB_TASKS = {'category_full':             get_category_full,
                   }
