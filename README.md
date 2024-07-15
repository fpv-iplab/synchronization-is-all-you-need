# Synchronization is All You Need:
## Exocentric-to-Egocentric Transfer for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs
This repository hosts the code related to the following paper:
Camillo Quattrocchi, Antonino Furnari, Daniele Di Mauro, Mario Valerio Giuffrida, Giovanni Maria Farinella: "Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs" (ECCV 2024). [Download](https://arxiv.org/pdf/2312.02638)

The code in this repository is based on this repository: [Repository](https://github.com/assembly-101/assembly101-temporal-action-segmentation)

If you use the code hosted in this repository, please cite the following paper: 
```
@article{quattrocchi2023synchronization,
  title={Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs},
  author={Quattrocchi, Camillo and Furnari, Antonino and Di Mauro, Daniele and Giuffrida, Mario Valerio and Farinella, Giovanni Maria},
  journal={arXiv preprint arXiv:2312.02638},
  year={2023}
}
```
## Contents
* ❓[Problem Definition](#problem-definition)
* ❗[Proposed Method](#proposed-method)
* 🥽[Data](#data)
* 🔃[Training](#training)
* 🏅[Evaluate](#evaluate)

## Problem Definition
![](https://github.com/fpv-iplab/synchronization-is-all-you-need/blob/main/assets/problem_definition.png?raw=true)
## Proposed Method
![](https://github.com/fpv-iplab/synchronization-is-all-you-need/blob/main/assets/proposed_method.png?raw=true)

## Data
Per-frame features are required as input. The features used in this work were extracted using DINOv2 (dinov2_vitl14, 1024-D). Link to the DINOv2 repository: [DINOv2](https://github.com/facebookresearch/dinov2/tree/main)

An example of code used to extract features is shown here: [DINOv2_feature_extractor.py](https://github.com/fpv-iplab/synchronization-is-all-you-need/blob/main/DINOv2_feature_extractor.py)

Run [data/data_stat.py](https://github.com/fpv-iplab/synchronization-is-all-you-need/blob/main/data/data_stat.py) to generate data statistics for each video.

## Training
To evaluate the trained models, first replace the model and feature paths within the `main_{oracle/transfrormer/distillation}.py`, `transformer_and_distillation.py` e `dataset_{oracle/distillation}.py` codes.

To train your model, run:
```
python main_{oracle/transfrormer/distillation}.py --action train --feature_path lmdb_path --split train
```
Or:
```
python transformer_and_distillation.py --action train --feature_path lmdb_path --split train
```

## Evaluate
To evaluate the trained models, first replace the model and feature paths within the `main_{oracle/transfrormer/distillation}.py`, `transformer_and_distillation.py` e `dataset_{oracle/distillation}.py` codes.

To evaluate the trained models:
```
python main_{oracle/transfrormer/distillation}.py --action predict --feature_path lmdb_path --test_aug 0
```
Or:
```
python transformer_and_distillation.py --action predict --feature_path lmdb_path --test_aug 0
```
