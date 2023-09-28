# MSSL: Multi-task Semi-supervised Learning for Vascular Network Segmentation and Renal Cell Carcinoma Classification (REMIA 2022)
***Best Poster Award*** \
<a href="https://link.springer.com/chapter/10.1007/978-3-031-16876-5_1"><img src="https://img.shields.io/badge/link.springer-10.1007-%23B31B1B"></a>
<a href="https://drive.google.com/file/d/1p42CPRfAgPPuY7_HrS-zVKJeXI88G0jS/view?usp=drive_link"><img src="https://img.shields.io/badge/Poster%20-online-brightgreen"></a>
<a href="https://drive.google.com/file/d/1q-uq_tS11zYJ84V2qT_-MDjccnh7lFmA/view?usp=drive_link"><img src="https://img.shields.io/badge/Presentation%20-online-brightgreen"></a>
<be>

## Introduction
<center>
<img src="https://github.com/medxiaorudan/MSSL/blob/main/images/model.png" width="700" > 
</center>

This is a PyTorch implementation of the paper Multi-task Semi-supervised Learning for Vascular Network Segmentation and Renal Cell Carcinoma Classification.

We propose an end-to-end MTL-SSL model performing joint SSL segmentation and classification tasks to segment the vascular network using both labeled and unlabeled data, which is robust and outperforms other popular SSL and supervised learning methods.   

## Installation
An example of installation is shown below:
```
git clone https://github.com/medxiaorudan/MSSL.git
cd MSSL
conda create -n MSSL python=3.8
conda activate MSSL
pip install -r requirements.txt
```
## Setup 
The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/RCC/`.
- Specify the output directory in `configs/your_env.yml`. All results will be stored under this directory.
- Please download the pre-trained weights [here](https://github.com/HRNet/HRNet-Image-Classification) for the HRNet backbone. 
The provided config files use an HRNet-18 backbone. Download the `hrnet_w18_small_model_v2.pth` and save it to the directory `./models/pretrained_models/`.

## Training
The configuration files to train the model can be found in the `configs/` directory. The model can be trained by running the following command:
If want run every model separately
```
python main.py --config_env configs/env.yml --config_exp configs/$DATASET/$MODEL.yml
```

If want batch running models
```
chmod +x script/Script.sh
./script/Script.sh train main_SSL_multi main_SSL_single
./script/Script.sh test test_SSL_multi test_SSL_single
```

## Evaluation
The code for post-precessing and evaluation details can be found in [Post-seg processing and evaluation](https://github.com/medxiaorudan/RCC-MSSL/blob/main/Post-seg_processing_and_evaluation).

```python
eval_final_10_epochs_only: True
```

## Citation
If you find this repo useful for your research, please consider citing the following works:
```
@inproceedings{xiao2022multi,
  title={Multi-Task Semi-Supervised Learning for Vascular Network Segmentation and Renal Cell Carcinoma Classification},
  author={Xiao, Rudan and Ambrosetti, Damien and Descombes, Xavier},
  booktitle={MICCAI Workshop on Resource-Efficient Medical Image Analysis},
  pages={1--11},
  year={2022},
  organization={Springer}
}

@article{
  author={S. Vandenhende and S. Georgoulis and W. Van Gansbeke and M. Proesmans and D. Dai and L. Van Gool},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Multi-Task Learning for Dense Prediction Tasks: A Survey}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3054719}}

@article{vandenhende2020mti,
  title={MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning},
  author={Vandenhende, Simon and Georgoulis, Stamatios and Van Gool, Luc},
  journal={ECCV2020},
  year={2020}
}

@InProceedings{MRK19,
  Author    = {Kevis-Kokitsi Maninis and Ilija Radosavovic and Iasonas Kokkinos},
  Title     = {Attentive Single-Tasking of Multiple Tasks},
  Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  Year      = {2019}
}

@article{pont2015supervised,
  title={Supervised evaluation of image segmentation and object proposal techniques},
  author={Pont-Tuset, Jordi and Marques, Ferran},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2015},
}
```
