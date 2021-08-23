import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class LatentFactorData(object):
    """
    Abstract class for datasets with known ground truth factors
    """
    def __init__(
        self,
        dataset_path,
        task_name,
        num_factors,
        sample_shape,
        c_names,
        task_fn,
    ):
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.num_factors = num_factors
        self.sample_shape = sample_shape
        self.c_names = c_names
        self.task_fn = task_fn
        self._has_generators = False

    def _load_x_c_data(self):
        raise NotImplementedError(
            "Need to implement code for loading input and concept data"
        )

    def get_concept_values(self):
        if not self._has_generators:
            self._get_generators()
        return self.n_c_vals_list

    def _get_generators(self, train_size=0.85, random_state=42):

        x_data, c_data = self._load_x_c_data()
        x_data, c_data, y_data = self.task_fn(x_data, c_data)
        x_data = x_data.astype(np.float32)
        self.n_classes = len(np.unique(y_data))
        self.n_c_vals_list = [
            len(np.unique(c_data[:, i])) for i in range(c_data.shape[1])
        ]

        # Create dictionary of
        # concept_id --> dictionary: consequtive_id : original_id
        self.cid_new_to_old = []
        for i in range(c_data.shape[1]):
            unique_vals, unique_inds = np.unique(
                c_data[:, i],
                return_inverse=True,
            )
            self.cid_new_to_old.append(
                {i : unique_vals[i] for i in range(len(unique_vals))}
            )
            c_data[:, i] = unique_inds

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.c_train,
            self.c_test,
        ) = train_test_split(
            x_data,
            y_data,
            c_data,
            train_size=train_size,
            random_state=random_state,
        )

        self.n_train_samples = self.c_train.shape[0]
        self.n_test_samples = self.c_test.shape[0]
        self.data_gen_train = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.c_train, self.y_train)
        )
        self.data_gen_test = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.c_test, self.y_test)
        )
        self._has_generators = True

    def load_data(self):
        return self.data_gen_train, self.data_gen_test, self.c_names



# ===========================================================================
#                   Task definition utility functions
# ===========================================================================

def built_task_fn(label_fn, filter_fn=None):
    def task_fn(x_data, c_data):
        return get_task_data(x_data, c_data, label_fn, filter_fn=filter_fn)
    return task_fn


def get_task_data(x_data, c_data, label_fn, filter_fn=None):

    if filter_fn is not None:
        ids = np.array([filter_fn(c) for c in c_data])
        ids = np.where(ids)[0]
        c_data = c_data[ids]
        x_data = x_data[ids]

    y_data = np.array([label_fn(c) for c in c_data])

    return x_data, c_data, y_data



