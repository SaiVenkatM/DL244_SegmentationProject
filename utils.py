import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")
from medpy.metric.binary import dc, ravd
from skimage import measure

# Function to get one-hot encoding for segmentation
def getOneHotSegmentation(batch):
    background_val = 0.
    labels = [1., 2., 3., 4., 5., 6., 7., 8.]

    one_hot_labels = torch.cat(
        [(batch == background_val)] + [(batch == label) for label in labels],
        dim=1)

    return one_hot_labels.float()

# Function to compute dice score for segmentation overlap
def dice_score(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label - 1).fill_(0).to(outputs.device)
    
    for label_num in range(1, max_label):
        iflat = (outputs == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = (iflat * tflat).sum()
        dice[label_num - 1] = (2. * intersection) / (iflat.sum() + tflat.sum())
    
    return dice

# Function to compute Hausdorff distance
def hd_updated(label_GT, label_CNN):
    seg = sitk.GetImageFromArray(label_CNN, isVector=False)
    reference_segmentation  = sitk.GetImageFromArray(label_GT, isVector=False)
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(reference_segmentation)
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    hausdorff_distance_filter.Execute(reference_segmentation, seg)
    hd_new = hausdorff_distance_filter.GetAverageHausdorffDistance()

    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(seg)

    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distances = list(sitk.GetArrayViewFromImage(seg2ref_distance_map)[sitk.GetArrayViewFromImage(seg2ref_distance_map) != 0])
    seg2ref_distances += [0] * (num_segmented_surface_pixels - len(seg2ref_distances))

    ref2seg_distances = list(sitk.GetArrayViewFromImage(ref2seg_distance_map)[sitk.GetArrayViewFromImage(ref2seg_distance_map) != 0])
    ref2seg_distances += [0] * (num_reference_surface_pixels - len(ref2seg_distances))

    all_surface_distances = seg2ref_distances + ref2seg_distances
    msd_new = np.mean(all_surface_distances)
    
    return hd_new

# Function to compute average Hausdorff distance
def avg_hd(imageDataCNN, imageDataGT, max_label):
    hd1 = np.zeros((max_label))
    for c_i in range(max_label):
        label_GT = (imageDataGT == c_i + 1).astype('uint8')
        label_CNN = (imageDataCNN == c_i + 1).astype('uint8')

        if np.count_nonzero(label_CNN) > 0 and np.count_nonzero(label_GT) > 0:
            hd1[c_i] = hd_updated(label_GT, label_CNN)
        elif np.count_nonzero(label_GT) > 0 and np.count_nonzero(label_CNN) == 0:
            hd1[c_i] = 2.
        elif np.count_nonzero(label_GT) == 0 and np.count_nonzero(label_CNN) > 0:
            hd1[c_i] = 2.
        elif np.count_nonzero(label_GT) == 0 and np.count_nonzero(label_CNN) == 0:
            hd1[c_i] = 0.

    return hd1

# Function to initialize weights
def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Dice Loss class
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, labels):
        numerator = 2 * torch.sum(labels * probs, dim=(2,3,4))
        denominator = torch.sum(labels + probs ** 2, dim=(2,3,4)) + self.smooth
        return 1 - torch.mean(numerator / denominator)

# Cross Entropy Loss class
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, net_output, target):
        return self.ce(net_output, target.long())

# Combined Dice and Cross Entropy Loss class
class DC_CE_Loss(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1):
        super(DC_CE_Loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()
        self.dc = DiceLoss()
    
    def forward(self, net_output, target_ce, target_dc):
        dc_loss = self.dc(net_output, target_dc) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_ce.long()) if self.weight_ce != 0 else 0
        return ce_loss + dc_loss

# Function to count trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to perform 3D affine augmentation
def augmentAffine(img_in, seg_in, strength=0.05):
    B, C, D, H, W = img_in.size()
    
    affine_matrix = (torch.eye(3, 4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)
    meshgrid = F.affine_grid(affine_matrix, torch.Size((B, 1, D, H, W)))
    img_out = F.grid_sample(img_in, meshgrid, padding_mode='border')
    seg_out = F.grid_sample(seg_in.float().unsqueeze(1), meshgrid, mode='nearest').long().squeeze(1)
    
    return img_out, seg_out
