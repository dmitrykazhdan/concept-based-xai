import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout


def small_cnn(input_shape, num_classes=10):
    '''
    CNN architecture used in CME (https://arxiv.org/abs/2010.13233) for the dSprites task
    :param input_shape: input sample shape
    :param num_classes: number of output classes
    :return: compiled keras model
    '''

    inputs = tf.keras.Input(shape=input_shape)
    x = get_cnn_body_layers(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def get_cnn_body_layers(inputs):
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2,
                               activation='relu', padding="same", name="e1")(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2,
                               activation='relu', padding="same", name="e2")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2,
                               activation='relu', padding="same", name="e3")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2,
                               activation='relu', padding="same", name="e4")(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    return x


def multi_task_cnn(input_shape, num_concpet_values, concept_names=[]):
    '''

    :param input_shape:
    :param num_concpet_values: list of containing number of concepts values for each concepts (e.g. [3,6,40,32,3] for dSprites)
    :return: CNN arrchitecture model with shared convolutional layers and multiple heads for respective task to predict
    multiple concept values in different outputs
     color, shape, scale, rotation, x and y positions
    '''

    if concept_names != []:
        assert len(
            num_concpet_values) == len(concept_names), "The number of concepts is different for the values and the names"
    inputs = tf.keras.Input(shape=input_shape)

    h = get_cnn_body_layers(inputs)

    outpuut_layers = []
    for i, c in enumerate(num_concpet_values):
        l = tf.keras.layers.Dense(num_concpet_values[i], activation="softmax", name="l" + str(i) if concept_names == [] else concept_names[
            i])(h)
        outpuut_layers.append(l)

    model = tf.keras.Model(inputs=inputs, outputs=outpuut_layers)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) for c in num_concpet_values],
                  metrics=['acc'])

    return model

def sigmoid_cnn(input_shape, num_concept_values, concept_names=[]):
    '''

    :param input_shape:
    :param num_concpet_values: list of containing number of concepts values for each concepts (e.g. [3,6,40,32,3] for dSprites)
    :return: CNN arrchitecture model with shared convolutional layers and multiple heads for respective task to predict
    multiple concept values in different outputs
     color, shape, scale, rotation, x and y positions
    '''

    if concept_names != []:
        assert len(
            num_concept_values) == len(concept_names), "The number of concepts is different for the values and the names"

    inputs = tf.keras.Input(shape=input_shape)

    h = get_cnn_body_layers(inputs)

    output_layers = []
    for i, c in enumerate(num_concept_values):
        l = tf.keras.layers.Dense(num_concept_values[i], activation="sigmoid", name="l" + str(i) if concept_names == [] else concept_names[
            i])(h)
        output_layers.append(l)

    model = tf.keras.Model(inputs=inputs, outputs=output_layers)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) for c in num_concept_values],
                  metrics=['acc'])

    return model

def conv_encoder(input_shape, num_latent):
    """CNN encoder architecture used in the 'Challenging Common Assumptions in the Unsupervised Learning
       of Disentangled Representations' paper (https://arxiv.org/abs/1811.12359)

       Note: model is uncompiled
    """

    inputs = tf.keras.Input(shape=input_shape)

    e1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2,
                                activation='relu', padding="same", name="e1")(inputs)

    e2 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2,
                                activation='relu', padding="same", name="e2")(e1)

    e3 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2,
                                activation='relu', padding="same", name="e3")(e2)

    e4 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2,
                                activation='relu', padding="same", name="e4")(e3)

    flat_e4 = tf.keras.layers.Flatten()(e4)
    e5 = tf.keras.layers.Dense(256, activation='relu', name="e5")(flat_e4)

    means = tf.keras.layers.Dense(num_latent, activation=None, name="means")(e5)
    log_var = tf.keras.layers.Dense(num_latent, activation=None, name="log_var")(e5)

    encoder = tf.keras.Model(inputs=inputs, outputs=[means, log_var])

    return encoder


def deconv_decoder(output_shape, num_latent):
    """CNN decoder architecture used in the 'Challenging Common Assumptions in the Unsupervised Learning
       of Disentangled Representations' paper (https://arxiv.org/abs/1811.12359)

       Note: model is uncompiled
    """

    latent_inputs = tf.keras.Input(shape=(num_latent,))
    d1 = tf.keras.layers.Dense(256, activation='relu')(latent_inputs)
    d2 = tf.keras.layers.Dense(1024, activation='relu')(d1)
    d2_reshaped = tf.keras.layers.Reshape([4, 4, 64])(d2)

    d3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2,
                                         activation='relu', padding="same")(d2_reshaped)

    d4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2,
                                         activation='relu', padding="same")(d3)

    d5 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2,
                                         activation='relu', padding="same")(d4)

    d6 = tf.keras.layers.Conv2DTranspose(filters=output_shape[2], kernel_size=4, strides=2, padding="same")(d5)
    output = tf.keras.layers.Reshape(output_shape)(d6)

    decoder = tf.keras.Model(inputs=latent_inputs, outputs=[output])

    return decoder
