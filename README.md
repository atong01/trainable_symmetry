<div align="center">

# Trainable Symmetry

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1911.06253-B31B1B.svg)](https://arxiv.org/abs/1911.06253)

<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

This is code for the paper \`\`Understanding Graph Neural Networks with Generalized Geometric Scattering Transforms''. For tables presented in the paper see `notebooks/results_eval.ipynb`.

## Description

This code implements a generalized geometric scattering transform implemented in pytorch and pytorch lightning and configured by hydra.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/atong01/trainable_symmetry
cd trainable_symmetry

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Configure directories in `.env` as needed.

To reproduce experiments in paper (also in `scripts/basic.sh`):

```bash
python src/train.py -m datamodule.transform_args.alpha=-0.5,-0.25,0.0,0.25,0.5 \
  datamodule.dataset=NCI1,NCI109,DD,PROTEINS,MUTAG,PTC_MR,REDDIT-BINARY,REDDIT-MULTI-5K,COLLAB,IMDB-BINARY,IMDB-MULTI \
  logger=wandb \
  datamodule.transform_args.power=1,2 \
  seed=0,1,2,3,4,5,6,7,8,9

python src/train.py -m datamodule.transform_args.alpha=-0.5,-0.25,0.0,0.25,0.5 \
  datamodule.dataset=NCI1,NCI109,DD,PROTEINS,MUTAG,PTC_MR,REDDIT-BINARY,REDDIT-MULTI-5K,COLLAB,IMDB-BINARY,IMDB-MULTI \
  logger=wandb \
  datamodule.transform_args.power=1 \
  +datamodule.transform_args.cheb_order=10,100\
  seed=0,1,2,3,4,5,6,7,8,9
```

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20
```

# BibTex Citation

```
@misc{perlmutter_understanding_2019,
  doi = {10.48550/ARXIV.1911.06253},
  url = {https://arxiv.org/abs/1911.06253},
  author = {Perlmutter, Michael and Gao, Feng and Wolf, Guy and Hirn, Matthew},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Understanding Graph Neural Networks with Asymmetric Geometric Scattering Transforms},
  publisher = {arXiv},
  year = {2019},
}
```
