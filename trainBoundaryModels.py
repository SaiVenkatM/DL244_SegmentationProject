import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import dill
import os
import nibabel as nib
import argparse

from utils import count_parameters, DiceLoss, CrossEntropyLoss, DC_CE_Loss, dice_score, getOneHotSegmentation, avg_hd, weights_init, augmentAffine, mtl_loss

from models.boundaryModel import unet3d_mtl_tsd, attention_unet_3d

def main():
    parser = argparse.ArgumentParser(description='Multi-task Learning Model Training')
    parser.add_argument('--e', default=300, type=int, help='Total number of epochs')
    parser.add_argument('--b', default=1, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--data_folder', default='../data', type=str, help='Root dataset path')
    parser.add_argument('--m', default='UNet', type=str, help='Model name')
    parser.add_argument('--conf', default='tsd', type=str, help='Model configuration')
    parser.add_argument('--aug', default=False, type=str, help='Data augmentation')
    parser.add_argument('--lambda_edge', default=1.0, type=float, help='Boundary loss weight')
    parser.add_argument('--output_folder', default='../data', type=str, help='Output folder for results')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer type')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    images, labels, edges = [], [], []

    num_samples = len(os.listdir(os.path.join(args.data_folder, 'images1')))
    data_indices = np.arange(num_samples) + 1

    for i in data_indices:
        img_path = os.path.join(args.data_folder, 'images1', f'pancreas_ct{i}.nii.gz')
        seg_path = os.path.join(args.data_folder, 'labels1', f'label_ct{i}.nii.gz')
        edge_path = os.path.join(args.data_folder, 'contours', f'label_ct{i}.nii.gz')

        img = nib.load(img_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()
        edge = nib.load(edge_path).get_fdata()

        seg[seg == 11] = 2
        seg[seg == 14] = 8

        images.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
        labels.append(torch.from_numpy(seg).unsqueeze(0).long())
        edges.append(torch.from_numpy(edge).unsqueeze(0))

    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    edges = torch.cat(edges, 0)

    num_labels = 9  # 8 organs + 1 background

    # Model selection and initialization
    model_map = {
        'unet': {'tsd': unet3d_mtl_tsd},
        'attenunet': {'tsd': attention_unet_3d}
    }

    net = model_map[args.model][args.conf]()
    net.apply(weights_init)
    print(f'{args.model} params: {count_parameters(net)}')
    net = nn.DataParallel(net).cuda()

    criterion = mtl_loss()

    # Optimizer selection
    optimizer_map = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'RMSprop': optim.RMSprop
    }
    optimizer = optimizer_map[args.optimizer](net.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    run_loss = np.zeros(args.epochs)

    fold_size = images.size(0)
    num_samples_for_eval = fold_size - fold_size % 4

    best_val_score = 0.0
    best_epoch = 0

    np.random.seed(1)
    shuffled_indices = np.random.permutation(num_samples)
    train_indices = shuffled_indices[:28]
    val_indices = shuffled_indices[28:32]
    test_indices = shuffled_indices[32:42]

    val_dice_scores = []
    segmentation_losses, classification_losses = [], []

    for epoch in range(args.epochs):
        net.train()
        run_loss[epoch] = 0.0
        epoch_start_time = time.time()
        np.random.seed(epoch)
        shuffled_train_indices = np.random.permutation(train_indices)
        epoch_indices = torch.from_numpy(shuffled_train_indices).view(args.batch, -1)

        for batch_indices in epoch_indices.split(1, dim=1):
            batch_indices = batch_indices.squeeze(1)
            
            with torch.no_grad():
                if args.aug:
                    augmented_images, label_segs = augmentAffine(images[batch_indices].cuda(), labels[batch_indices].cuda(), strength=0.075)
                    label_edges = edges[batch_indices].cuda()
                else:
                    augmented_images, label_segs, label_edges = images[batch_indices].cuda(), labels[batch_indices].cuda(), edges[batch_indices].cuda()

                torch.cuda.empty_cache()

            label_segs_onehot = getOneHotSegmentation(label_segs.unsqueeze(1))
            label_edges = label_edges.unsqueeze(1)
            optimizer.zero_grad()
            predictions, predicted_edges = net(augmented_images)
            total_loss, seg_loss, class_loss = criterion(F.softmax(predictions, dim=1), label_segs_onehot, predicted_edges, label_edges, args.lambda_edge)
            segmentation_losses.append(seg_loss.cpu().detach().numpy())
            classification_losses.append(class_loss.cpu().detach().numpy())

            total_loss.backward()
            run_loss[epoch] += total_loss.item()
            optimizer.step()
            del total_loss, predictions, augmented_images, label_segs_onehot
            torch.cuda.empty_cache()

        scheduler.step()

        # Validation
        net.eval()
        validation_dice_scores = []
        for val_idx in val_indices:
            with torch.no_grad():
                val_img = images[val_idx].unsqueeze(1).cuda()
                val_pred, _ = net(val_img)
                val_pred = F.softmax(val_pred, dim=1).argmax(dim=1)
                dice_score_val = dice_score(val_pred.cpu(), labels[val_idx].unsqueeze(1), num_labels)
                validation_dice_scores.append(dice_score_val.cpu().numpy())
        
        mean_dice_score = np.nanmean(validation_dice_scores, axis=0)
        mean_dice_score_percent = mean_dice_score * 100.0

        print(f'Epoch {epoch}: Time={time.time() - epoch_start_time:.3f}s, Total Loss={run_loss[epoch]:.4f}, Segmentation Loss={np.mean(segmentation_losses):.4f}, Classification Loss={np.mean(classification_losses):.4f}')
        print(f'Validation Mean Dice Score: {mean_dice_score_percent}')
        print(f'Organ-wise Validation Dice Scores: {mean_dice_score_percent}')

        if mean_dice_score_percent > best_val_score:
            best_val_score = mean_dice_score_percent
            best_epoch = epoch
            torch.save(net.state_dict(), os.path.join(args.output_folder, f"Best_{args.model}_{args.conf}.pth"), pickle_module=dill)
            print('Model saved successfully!')

        val_dice_scores.append(mean_dice_score_percent)

    print(f'Highest validation dice score: {best_val_score:.3f} at epoch: {best_epoch}')

    # Model inference
    del net
    if args.model == 'unet':
        net = model_map['unet'][args.conf]()
    elif args.model == 'attenunet':
        net = model_map['attenunet'][args.conf]()

    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(os.path.join(args.output_folder, f"Best_{args.model}_{args.conf}.pth")))
    net.eval()

    test_dice_scores, hausdorff_distances = [], []

    for test_idx in test_indices:
        with torch.no_grad():
            test_img = images[test_idx].unsqueeze(1).cuda()
            test_pred, _ = net(test_img)
            test_pred = F.softmax(test_pred, dim=1).argmax(dim=1)
            dice_score_test = dice_score(test_pred.cpu(), labels[test_idx].unsqueeze(1), num_labels)
            hausdorff_dist = avg_hd(test_pred.squeeze(0).cpu().numpy(), labels[test_idx].numpy(), num_labels - 1)

            test_dice_scores.append(dice_score_test.numpy())
            hausdorff_distances.append(hausdorff_dist)

            nifti_img = nib.Nifti1Image(test_pred.cpu().numpy().squeeze(0), np.eye(4))
            nifti_save_path = os.path.join(args.output_folder, 'nifti_preds', f'{test_idx}.nii.gz')
            os.makedirs(os.path.dirname(nifti_save_path), exist_ok=True)
            nib.save(nifti_img, nifti_save_path)

    mean_test_dice_score = np.mean(test_dice_scores, axis=0)
    mean_hausdorff_distance = np.mean(hausdorff_distances, axis=0)

    print(f'Test Mean Dice Score: {mean_test_dice_score * 100.0}')
    print(f'Test Mean Hausdorff Distance: {mean_hausdorff_distance}')

    np.save(os.path.join(args.output_folder, 'test_results', 'mean_test_dice_score.npy'), mean_test_dice_score)
    np.save(os.path.join(args.output_folder, 'test_results', 'mean_hausdorff_distance.npy'), mean_hausdorff_distance)

if __name__ == '__main__':
    main()
