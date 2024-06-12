# 3D abdominal multi-organ segmentation


This repository contains pytorch implementation corresponding to single-task and multi-task models.

## 3D baseline models:
1. 3D UNet
2. 3D Attention-UNet

## 3D boundary-constrained models3. 3D UNet-MTL-TSD
1. 3D UNet++-MT-D
2. 3D Attention-UNet-MT-D


## Downloading the datasets:

First download the datasets from links below:
1. BTCV dataset {For images: Use [Link1](https://www.synapse.org/#!Synapse:syn3193805) , For labels: Use [Link2](https://zenodo.org/record/1169361#.YnIytuhBw2w)}

## Dataset preparation:
1. To prepare the data, we use the pipeline utilized in [Obelisk-Net](https://www.sciencedirect.com/science/article/abs/pii/S136184151830611X) paper
2. Use the ITK-snap to crop the scans and labels according to the bounding box coordinates given on [Link2](https://zenodo.org/record/1169361#.YnIytuhBw2w).

## Dataset organization
### Organization of Data for training baseline models:
Organize the CT scans and their corresponding labels according to the format below:
```
Data Folder:
     --data:
            --images1:
                     --pancreas_ct1.nii.gz
                     --pancreas_ct2.nii.gz
                     .....................
            --labels1:
                     --label_ct1.nii.gz
                     --label_ct2.nii.gz
                     .....................
```
### Organization of data for training boundary models:
```
Data Folder:
     --data:
            --images1:
                     --pancreas_ct1.nii.gz
                     --pancreas_ct2.nii.gz
                     .....................
            --labels1:
                     --label_ct1.nii.gz
                     --label_ct2.nii.gz
                     .....................
            --edges:
                     --label_ct1.nii.gz
                     --label_ct2.nii.gz
                     .....................
```
## Model Training 
To train the baseline models, use the following command:

`python trainBaselineModels.py --b 4 --e 100 --m unet --lr 0.001 --data_folder 'path where the data is stored' --output_folder 'path to save the results'`

To train the boundary-constrained models, use the following command:

`python trainBoundaryModels.py --b 4 --e 300 --m unet --lr 0.001 --lambda_edge 0.5 --data_folder 'path where the data is stored' --output_folder 'path to save the results'
`

The Models have been trained using Nvidia L4 GPU, you will need to reduce the size of the batch if you use one GPU. 