import matplotlib.pyplot as plt

import numpy as np

# Update this to change font sizes
plt.rcParams.update({'font.size': 18})


def visualize_nearest_neighbours(f_data, x_data, n_concepts, topic_model, n_prototypes=5):

    fig, axes                   = plt.subplots(n_concepts, n_prototypes, figsize=(20, 20))
    dim1_size, dim2_size        = f_data.shape[1], f_data.shape[2]
    dim1_factor, dim2_factor    = int(x_data.shape[1] / dim1_size), int(x_data.shape[2] / dim2_size)

    f_data_n    = f_data / (np.linalg.norm(f_data,axis=3,keepdims=True)+1e-9)
    topic_vec   = topic_model.topic_vector.numpy()
    topic_vec_n = topic_vec / (np.linalg.norm(topic_vec, axis=0, keepdims=True) + 1e-9)
    topic_prob  = np.matmul(f_data_n, topic_vec_n)

    for i in range(n_concepts):
        # Note: topic_prob is an array of shape (n_samples, kernel_w, kernel_h, n_concepts)
        # Here, we retrieve the indices of the n_imgs_per_c largest elements
        ind = np.argsort(-topic_prob[:,:,:,i].flatten())[:n_prototypes]

        for jc, j in enumerate(ind):
            # Retrieve the dim0 index
            dim0 = int(np.floor(j/(dim1_size*dim2_size)))
            # Retrieve the dim1 index
            dim1 = int((j-dim0*(dim1_size*dim2_size))/dim2_size)
            # Retrieve the dim2 index
            dim2 = int((j-dim0*(dim1_size*dim2_size))%dim2_size)

            # Retrieve the subpart of the image corresponding to the activated patch
            dim1_start = dim1_factor*dim1
            dim2_start = dim2_factor*dim2
            img = x_data[dim0, :, :, :]
            img = img[dim1_start:dim1_start+dim1_factor, dim2_start:dim2_start+dim2_factor, :]

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
