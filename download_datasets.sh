#!/bin/bash
set -e

# Note: code partially-adapted from the
#       'https://github.com/google-research/disentanglement_lib' repo


echo "Downloading small_norb."
if [[ ! -d "small_norb" ]]; then
  mkdir small_norb
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat" ]]; then
  wget -O small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
  gunzip small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat" ]]; then
  wget -O small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
  gunzip small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-info.mat" ]]; then
  wget -O small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
  gunzip small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat" ]]; then
  wget -O small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
  gunzip small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-testing-cat.mat" ]]; then
  wget -O small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
  gunzip small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat" ]]; then
  wget -O small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
  gunzip small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
fi
echo "Downloading small_norb completed!"


echo "Downloading cars dataset."
if [[ ! -d "cars" ]]; then
  wget -O nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
  tar xzf nips2015-analogy-data.tar.gz
  rm nips2015-analogy-data.tar.gz
  mv data/cars .
  rm -r data
fi
echo "Downloading cars completed!"


echo "Downloading dSprites dataset."
if [[ ! -d "dsprites" ]]; then
  mkdir dsprites
  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
fi
echo "Downloading dSprites completed!"


echo "Downloading shapes3d dataset."
if [[ ! -d "shapes3d" ]]; then
  mkdir shapes3d
  wget -O shapes3d/3dshapes.h5 https://storage.cloud.google.com/3d-shapes/3dshapes.h5

fi
echo "Downloading shapes3d completed!"