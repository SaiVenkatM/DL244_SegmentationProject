import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import dill
import os
import nibabel as nib
import argparse

from utils import count_parameters, DiceLoss, CrossEntropyLoss, DC_CE_Loss, dice_score, getOneHotSegmentation, avg_hd, weights_init, augmentAffine
from models.baselineModel import UNet3D, ModifiedUNet3D

def train_baseline_model():
    parser = argparse.ArgumentParser(description='Baseline Model Training')
    parser.add_argument('--e', default=300, type=int, help='Total number of epochs')
    parser.add_argument('--b', default=1, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--data_folder', default='../data', type=str, help='Path to dataset root')
    parser.add_argument('--m', default='unet', type=str, choices=['unet', 'unetplus', 'attenunet'], help='Model type')
    parser.add_argument('--aug', action='store_true', help='Enable data augmentation')
    parser.add_argument('--output_folder', default='../data', type=str, help='Output folder for results')
    parser.add_argument('--loss', default='dice_loss', type=str, choices=['dice_loss', 'CE_loss', 'DC_CE_loss'], help='Loss function')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'RMSprop'], help='Optimizer')

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Load images and segmentations
    images, segmentations = [], []
    total_data = len(os.listdir(os.path.join(args.data_folder, 'images1')))
    data_indices = np.arange(total_data) + 1

    for index in data_indices:
        image_path = os.path.join(args.data_folder, 'images1', f'pancreas_ct{index}.nii.gz')
        seg_path = os.path.join(args.data_folder, 'labels1', f'label_ct{index}.nii.gz')
        image = nib.load(image_path).get_fdata()
        segmentation = nib.load(seg_path).get_fdata()
        segmentation[segmentation == 11] = 2.
        segmentation[segmentation == 14] = 8.

        images.append(torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float())
        segmentations.append(torch.from_numpy(segmentation).unsqueeze(0).long())

    images = torch.cat(images, 0) / 1024.0 + 1.0
    segmentations = torch.cat(segmentations, 0)
    num_labels = 9

    # Initialize model
    if args.model == "unet":
        model = UNet3D()
    elif args.model == "attenunet":
        model = ModifiedUNet3D()

    model.apply(weights_init)
    print(f'{args.model} parameters: {count_parameters(model)}')
    model = nn.DataParallel(model).cuda()

    # Define loss function
    if args.loss == "dice_loss":
        criterion = DiceLoss()
    elif args.loss == "CE_loss":
        criterion = CrossEntropyLoss()
    elif args.loss == "DC_CE_loss":
        criterion = DC_CE_Loss()

    # Define optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    training_loss = np.zeros(args.epochs)

    data_size = images.size(0)
    data_size4 = data_size - data_size % 4
    best_val_score = 0.0
    best_epoch = 0

    np.random.seed(1)
    epoch_indices = np.random.permutation(total_data)
    train_indices = epoch_indices[:28]
    val_indices = epoch_indices[28:32]
    test_indices = epoch_indices[32:42]
    print('epoch_indices:', epoch_indices)
    print('val_indices:', val_indices)
    print('test_indices:', test_indices)
    val_dice_scores = []

    for epoch in range(args.epochs):
        model.train()
        training_loss[epoch] = 0.0
        start_time = time.time()
        np.random.seed(epoch)
        epoch_permutation = np.random.permutation(train_indices)
        epoch_permutation = torch.from_numpy(epoch_permutation).view(args.batch, -1)

        for iteration in range(epoch_permutation.size(1)):
            batch_indices = epoch_permutation[:, iteration]

            with torch.no_grad():
                if args.aug:
                    batch_images, batch_labels = augmentAffine(images[batch_indices].cuda(), segmentations[batch_indices].cuda(), strength=0.075)
                else:
                    batch_images, batch_labels = images[batch_indices].cuda(), segmentations[batch_indices].cuda()

                torch.cuda.empty_cache()
            batch_labels_dc = torch.unsqueeze(batch_labels, 1)
            batch_labels_dc = getOneHotSegmentation(batch_labels_dc)
            optimizer.zero_grad()
            predictions = model(batch_images)
            loss = criterion(F.softmax(predictions, dim=1), batch_labels_dc)
            loss.backward()
            training_loss[epoch] += loss.item()
            optimizer.step()
            del loss, predictions, batch_images, batch_labels_dc
            torch.cuda.empty_cache()

        scheduler.step()
        train_time = time.time() - start_time
        print(f'Epoch {epoch}, Training Time: {train_time:.3f}s, Total Loss: {training_loss[epoch]:.4f}')

        # Validation
        model.eval()
        val_dices = []
        for val_idx in val_indices:
            start_val_time = time.time()
            with torch.no_grad():
                val_image = images[val_idx].unsqueeze(0).cuda()
                val_prediction = model(val_image)
                val_argmax = torch.max(F.softmax(val_prediction, dim=1), dim=1)[1]
                if epoch == 0:
                    print(f'Time per validation image: {time.time() - start_val_time:.3f}s')

                dice_all = dice_score(val_argmax.cpu(), segmentations[val_idx].unsqueeze(0), num_labels)
                val_dices.append(dice_all.cpu().numpy())
                torch.cuda.empty_cache()

        val_mean_dice = np.nanmean(val_dices, axis=0)
        mean_dice_score = np.nanmean(val_mean_dice) * 100.0
        print(f'Mean Validation Dice: {mean_dice_score:.2f}')
        print(f'Organ Validation Dice: {val_mean_dice * 100.0}')

        if mean_dice_score > best_val_score:
            best_val_score = mean_dice_score
            best_epoch = epoch
            best_val_dice = val_mean_dice
            torch.save(model.state_dict(), os.path.join(args.output_folder, f"Best_{args.model}.pth"), pickle_module=dill)
            print('********** Model Saved Successfully **********')
        val_dice_scores.append(mean_dice_score)
    print(f'Best Validation Dice: {best_val_score:.2f} at Epoch {best_epoch}')

    # Inference
    del model
    if args.model == "unet":
        model = UNet3D()
    elif args.model == "attenunet":
        model = ModifiedUNet3D()

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(args.output_folder, f"Best_{args.model}.pth")))
    model.eval()

    test_dices, hausdorff_distances = [], []
    for test_idx in test_indices:
        with torch.no_grad():
            test_image = images[test_idx].unsqueeze(0).cuda()
            test_prediction = model(test_image)
            test_argmax = torch.max(F.softmax(test_prediction, dim=1), dim=1)[1]
            test_argmax_np = test_argmax.cpu().numpy().squeeze(0)

            affine = np.eye(4) * 2
            nifti_img = nib.Nifti1Image(test_argmax_np, affine)
            nifti_dir = os.path.join(args.output_folder, 'nifti_preds')
            os.makedirs(nifti_dir, exist_ok=True)
            nifti_path = os.path.join(nifti_dir, f'{test_idx}.nii.gz')
            nib.save(nifti_img, nifti_path)
            torch.cuda.synchronize()

            dice_all = dice_score(test_argmax.cpu(), segmentations[test_idx].unsqueeze(0), num_labels)
            hd = avg_hd(test_argmax_np, segmentations[test_idx].numpy(), num_labels)

            test_dices.append(dice_all.numpy())
            hausdorff_distances.append(hd)

    mean_test_dice = np.nanmean(test_dices, axis=0)
    mean_hausdorff = np.mean(hausdorff_distances, axis=0)
    mean_test_dice_score = np.nanmean(test_dices) * 100.0

    print(f'Mean Test Dice: {mean_test_dice_score:.2f}')
    print(f'Organ Test Dice: {mean_test_dice * 100.0}')

    test_results = {
        'mean_dice': mean_test_dice,
        'mean_hd': mean_hausdorff,
        'test_dices': test_dices,
        'hausdorff_distances': hausdorff_distances,
    }

    test_results_dir = os.path.join(args.output_folder, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    np.save(os.path.join(test_results_dir, 'test_results.npy'), test_results)

if __name__ == '__main__':
    train_baseline_model()
