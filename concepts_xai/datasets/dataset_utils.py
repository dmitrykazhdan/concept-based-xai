import numpy as np

from . import dSprites, shapes3D, smallNorb, cars3D

'''
A primer on bases:
Assume you have a vector A = (x, y, z), where every dimension is in base (a, b, c)
Then, in order to convert each of those dimensions to decimal, we do:

D = (z * 1) + (y * c) + (x * b * c)

Or, can define bases vector B = [b*c, c, 1], and then define D = B.A

Example:
(1, 0, 1) in bases (2, 2, 2) (i.e. in binary):
D = (2^0 * 1) + (2^1 * 0) + (2^2 * 1) = 1 + 4 = 5 
'''

def get_latent_bases(latent_sizes):
    return np.concatenate((latent_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))


# Convert a concept-based index into a single index
def latent_to_index(latents, latents_bases):
    return np.dot(latents, latents_bases).astype(int)

dataset_concept_names = {
    "dSprites":     dSprites.DSPRITES_concept_names,
    "shapes3d":     shapes3D.SHAPES3D_concept_names,
    "smallNorb":    smallNorb.SMALLNORB_concept_names,
    "cars3D":       cars3D.CARS_concept_names
}