# Concept-based XAI Library

CXAI is an open-source library for research on concept-based Explainable AI (XAI). 

CXAI supports a variety of different models, datasets, and evaluation metrics, associated with concept-based approaches:


### High-level Specs:

_Methods_: 
- [Now You See Me (CME): Concept-based Model Extraction](https://arxiv.org/abs/2010.13233)
- [Concept Bottleneck Models](https://arxiv.org/abs/2007.04612)
- [Weakly-Supervised Disentanglement Without Compromises](https://arxiv.org/abs/2002.02886)
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359)
- [Concept Whitening for Interpretable Image Recognition](https://arxiv.org/abs/2002.01650) (note: this approach has not yet been tested)
- [On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://arxiv.org/abs/1910.07969) (note: this approach has not yet been tested)


_Datasets_:
- [dSprites](https://github.com/deepmind/dsprites-dataset)
- [Shapes3D](https://github.com/deepmind/3d-shapes)
- [Cars3D](https://papers.nips.cc/paper/2015/hash/e07413354875be01a996dc560274708e-Abstract.html)
- [SmallNorb](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)

to get the datasets run script datasets/download_datasets.sh

### Requirements:

- Python 3.7 - 3.8
- See 'requirements.txt' for the rest


### Directories:

- Methods: defines the concept-based methods. Note: SSCC defines wrappers around these methods, that turn then into semi-supervised concept labelling methods.
- Experiments: experimental setups (To-be-added soon)
- Evaluation: different evaluation metrics to use for evaluating the semi-supervised concept labelling methods
- Datasets: datasets to use, including task functions


### Citing

If you find this code useful in your research, please consider citing:

```
@article{kazhdan2021disentanglement,
  title={Is Disentanglement all you need? Comparing Concept-based \& Disentanglement Approaches},
  author={Kazhdan, Dmitry and Dimanov, Botty and Terre, Helena Andres and Jamnik, Mateja and Li{\`o}, Pietro and Weller, Adrian},
  journal={arXiv preprint arXiv:2104.06917},
  year={2021}
}
```

This work has been presented at the [RAI](https://sites.google.com/view/rai-workshop/), [WeaSuL](https://weasul.github.io/), and [RobustML](https://sites.google.com/connect.hku.hk/robustml-2021/home) workshops, at [The Ninth International Conference on Learning Representations (ICLR 2021)](https://iclr.cc/).
