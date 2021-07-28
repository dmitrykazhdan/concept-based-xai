import os
import shutil

from .architectures import multi_task_cnn, small_cnn
from .model_loader import get_model


def tf_data_split(ds, test_size=0.15, n_samples=None):
    '''
    Method for train/test splitting a tf.dataset, assuming it was generated from numpy arrays
    :param ds: tf.dataset, assumed to be generated "from_tensor_slices" (see ./datasets/dSprites.py for an example)
    :param test_size: Ratio of test samples
    :param n_samples: Total number of samples in ds
    :return: Two tf datasets, obtained by splitting ds
    '''

    if n_samples is None:
        # Compute total number of samples, if not provided
        n_samples = int(ds.cardinality().numpy())

    # Split the dataset
    train_size            = int((1. - test_size) * n_samples)
    test_size             = n_samples - train_size
    ds_train              = ds.take(train_size)
    ds_test               = ds.skip(train_size).take(test_size)

    return ds_train, ds_test


def setup_experiment_dir(dir_path, overwrite=False):

    # Remove prior contents
    if overwrite and os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def convert_to_multioutput(c):
    multi_output = tuple([c[i:i + 1] for i in range(c.shape[0])])
    return multi_output


def setup_basic_model(dataset, n_epochs, save_path, model_type=""):
    '''
    Train a standard CNN model (used for CBM and CME)
    :param dataset: dataset class to train with
    :param n_epochs: number of epochs to train for
    :param save_path: path to saving/loading the model
    :param model_type: whether single_output ("") or multi_task
    :return:
    '''

    data_gen_train, data_gen_test, c_names = dataset.load_data()

    n_classes = dataset.n_classes
    input_shape = dataset.sample_shape
    n_samples = dataset.n_train_samples
    num_concpet_values = dataset.n_c_vals_list
    concept_names = dataset.c_names

    print(f"No. training samples: {n_samples}")
    # Train/Load CNN model using data generators
    if model_type == "multi_task":

        # remove y and convert concept data to multi-task
        data_gen_train_c = data_gen_train.map(lambda x, c, y: (x, convert_to_multioutput(c)))
        data_gen_t, data_gen_v = tf_data_split(data_gen_train_c, test_size=0.15, n_samples=n_samples)
        untrained_model = multi_task_cnn(input_shape, num_concpet_values, concept_names)
        #         data_gen_t,data_gen_v = data_gen_train_c,data_gen_train_c
        basic_model = get_model(untrained_model, save_path, data_gen_t, data_gen_v, epochs=n_epochs)
        results = basic_model.evaluate(data_gen_train_c.batch(batch_size=256))
    else:
        data_gen_train_no_c = data_gen_train.map(lambda x, c, y: (x, y))  # Remove concept data
        # Train/Load CNN model using data generators
        data_gen_t, data_gen_v = tf_data_split(data_gen_train_no_c, test_size=0.15, n_samples=n_samples)
        untrained_model = small_cnn(input_shape, n_classes)
        basic_model = get_model(untrained_model, save_path, train_gen=data_gen_t, val_gen=data_gen_v, epochs=n_epochs)

        # Evaluate the model
        data_gen_test_no_c = data_gen_test.map(lambda x, c, y: (x, y))
        results = basic_model.evaluate(data_gen_test_no_c.batch(batch_size=256))
    print("Model performance: ", results)

    return basic_model