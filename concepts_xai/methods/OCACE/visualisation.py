import matplotlib.pyplot as plt

import numpy as np

# Update this to change font sizes
plt.rcParams.update({'font.size': 18})


def visualize_nearest_neighbours(
    f_data,
    x_data,
    topic_model,
    n_prototypes=5,
    channels_axis=-1,
):

    topic_vec = topic_model.topic_vector.numpy()
    fig, axes = plt.subplots(
        topic_vec.shape[1],
        n_prototypes,
        figsize=(20, 20),
    )
    topic_prob = topic_model.concept_scores(f_data)
    if len(f_data.shape):
        # Then let's trivially expand its dimensions so that we always
        # work with 3D representations
        f_data = np.expand_dims(
            np.expand_dims(f_data, axis=1),
            axis=1,
        )
        topic_prob = np.expand_dims(
            np.expand_dims(topic_prob, axis=1),
            axis=1,
        )
    dim1_size, dim2_size = f_data.shape[1], f_data.shape[2]
    dim1_factor = int(x_data.shape[1] / dim1_size)
    dim2_factor = int(x_data.shape[2] / dim2_size)

    for i in range(topic_prob.shape[channels_axis]):
        # Then topic_prob is an array of shape
        # # (n_samples, kernel_w, kernel_h, n_concepts)
        # Here, we retrieve the indices of the n_imgs_per_c largest elements
        ind = np.argsort(-topic_prob[:, :, :, i].flatten())[:n_prototypes]

        for jc, j in enumerate(ind):
            # Retrieve the dim0 index
            dim0 = int(np.floor(j/(dim1_size*dim2_size)))
            # Retrieve the dim1 index
            dim1 = int((j-dim0 * (dim1_size*dim2_size))/dim2_size)
            # Retrieve the dim2 index
            dim2 = int((j-dim0 * (dim1_size*dim2_size)) % dim2_size)

            # Retrieve the subpart of the image corresponding to the
            # activated patch
            dim1_start = dim1_factor * dim1
            dim2_start = dim2_factor * dim2
            img = x_data[dim0, :, :, :]
            img = img[
                dim1_start: dim1_start + dim1_factor,
                dim2_start: dim2_start + dim2_factor,
                :,
            ]

            # Plot the image
            ax = axes[i, jc]

            if img.shape[-1] == 1:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.yaxis.grid(False)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

    fig.tight_layout()
    plt.show()
