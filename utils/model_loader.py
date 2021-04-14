import os
import tensorflow as tf


def get_model(model, model_save_path, overwrite=False, train_gen=None, val_gen=None, batch_size=256, epochs=250):
    '''
    Code for loading/training a given model
    :param model: Compiled Keras model
    :param model_save_path: Path for saving/loading the model weights
    :param train_gen: tf.dataset generator for training the model
    :param val_gen:   tf.dataset generator for validating the model
    :param batch_size: Batch size used during training
    :param epochs:     Number of epochs to use for training
    :return: Trained/loaded model
    '''

    if (overwrite) or (not os.path.exists(model_save_path)):

        if (train_gen is None) or (val_gen is None):
            raise ValueError("Training and/or Validation data generators not provided.")

        # Train model
        callbacks = []

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, verbose=True,
                                                         save_best_only=True, monitor='val_acc',
                                                         mode='auto', save_freq='epoch')
        callbacks.append(cp_callback)

        model.fit(train_gen.batch(batch_size), epochs=epochs,
                  validation_data=val_gen.batch(batch_size), callbacks=callbacks)

        #make sure to save the path
        dir_name = os.path.dirname(model_save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model.save_weights(model_save_path)

    else:
        print("Loading pre-trainined model")
        model.load_weights(model_save_path)

    return model