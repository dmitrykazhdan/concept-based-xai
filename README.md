# Concept-based XAI Library

CXAI is an open-source library for research on concept-based Explainable AI
(XAI).

CXAI supports a variety of different models, datasets, and evaluation metrics,
associated with concept-based approaches:


### High-level Specs:

_Methods_:
- [Now You See Me (CME): Concept-based Model Extraction](https://arxiv.org/abs/2010.13233).
- [Concept Bottleneck Models](https://arxiv.org/abs/2007.04612)
- [Weakly-Supervised Disentanglement Without Compromises](https://arxiv.org/abs/2002.02886)
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359)
- [Concept Whitening for Interpretable Image Recognition](https://arxiv.org/abs/2002.01650)
- [On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://arxiv.org/abs/1910.07969)
- [Towards Robust Interpretability with Self-Explaining Neural Networks
](https://arxiv.org/abs/1806.07538)


_Datasets_:
- [dSprites](https://github.com/deepmind/dsprites-dataset)
- [Shapes3D](https://github.com/deepmind/3d-shapes)
- [Cars3D](https://papers.nips.cc/paper/2015/hash/e07413354875be01a996dc560274708e-Abstract.html)
- [SmallNorb](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)

to get the datasets run script datasets/download_datasets.sh

### Requirements

- Python 3.7 - 3.8
- See 'requirements.txt' for the rest of required packages

### Installation
If installing from the source, please proceed by running the following command:
```bash
python setup.py install
```
This will install the `concepts-xai` package together with all its dependencies.

To test that the package has been successfully installed, you may run:
```python
import concepts_xai
help("concepts_xai")
```
to display all the subpackages included from this installation.

### Subpackages

- `datasets`: datasets to use, including task functions.
- `evaluation`: different evaluation metrics to use for evaluating our methods.
- `experiments`: experimental setups (To-be-added soon)
- `methods`: defines the concept-based methods. Note: SSCC defines wrappers around these methods, that turn then into semi-supervised concept labelling methods.
- `utils`: contains utility functions for model creation as well as data management.


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

This work has been presented at the [RAI](https://sites.google.com/view/rai-workshop/), [WeaSuL](https://weasul.github.io/), and
[RobustML](https://sites.google.com/connect.hku.hk/robustml-2021/home) workshops, at [The Ninth International Conference on Learning
Representations (ICLR 2021)](https://iclr.cc/).
