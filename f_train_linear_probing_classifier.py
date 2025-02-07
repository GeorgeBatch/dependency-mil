# built-in imports
import argparse
import datetime
import glob
import json
import os
import sys
from functools import partial

import numpy as np
import torch
import wandb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

from configs.config import Config
from source.data.datamodule_detailed import LungSubtypingSlideEmbeddingDM
from source.losses import custom_cross_entropy_loss
from source.metrics import multi_label_roc
from source.utils.general import seed_everything, load_object


def train(
        train_dataloader,
        model,
        criterion,
        optimizer,
        config,
        device: torch.device
):
    model.train()
    total_loss = 0

    for batch_index, batch in enumerate(train_dataloader):
        if 1 < config.limit_num_batches == batch_index:
            break

        optimizer.zero_grad()

        # ipdb.set_trace()

        # put on device in the dataset class
        bag_feats, bag_label, label_weight_mask = batch

        # ipdb.set_trace()

        # calculate bag predictions
        bag_prediction = model(bag_feats)  # bag_prediction: (batch_size, num_classes)

        # ipdb.set_trace()

        # compute loss
        loss = criterion(inputs_list=[bag_prediction], targets=bag_label, weights=label_weight_mask)

        # ipdb.set_trace()

        # backpropagate
        loss.backward()
        optimizer.step()

        # ipdb.set_trace()

        total_loss = total_loss + loss.item()
        sys.stdout.write(
            '\r \t Training bag [%d/%d] bag loss: %.4f' % (batch_index, len(train_dataloader), loss.item()))

    # division by len(train_dataloader) or len(train_dataloader.dataset) depends on whether the loss is averaged or summed (respectively) within each batch
    return total_loss / len(train_dataloader)


def evaluate(
        eval_dataloader: DataLoader,
        model,
        criterion,
        mode: str,
        saved_thresholds,
        config,
        device: torch.device
):
    model.eval()
    total_loss = 0
    test_labels = []
    test_label_weight_masks = []
    test_predictions = []
    # Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for batch_index, batch in enumerate(eval_dataloader):
            if 1 < config.limit_num_batches == batch_index:
                break

            # ipdb.set_trace()

            # put on device in the dataset class
            bag_feats, bag_label, label_weight_mask = batch

            # ipdb.set_trace()

            # calculate bag prediction
            bag_prediction = model(bag_feats)  # bag_prediction: (batch_size, num_classes)

            # ipdb.set_trace()

            # compute loss
            loss = criterion(inputs_list=[bag_prediction], targets=bag_label, weights=label_weight_mask)

            # ipdb.set_trace()

            # increment total loss
            total_loss = total_loss + loss.item()

            # ipdb.set_trace()

            sys.stdout.write(
                '\r \t Evaluating bag [%d/%d] bag loss: %.4f' % (batch_index, len(eval_dataloader), loss.item()))

            test_labels.extend([bag_label.cpu().numpy()])
            test_label_weight_masks.extend([label_weight_mask.cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).cpu().numpy()])

    # inside each batch, loss is averaged over the number of samples in the batch, so we divide by the number of batches
    avg_loss_per_sample = total_loss / len(eval_dataloader)

    # ipdb.set_trace()

    # before np.vstack, both test_labels and test_predictions are lists of arrays of size (batch_size, num_classes)
    test_labels = np.vstack(test_labels)
    test_predictions = np.vstack(test_predictions)
    test_label_weight_masks = np.vstack(test_label_weight_masks)
    assert test_labels.shape == test_predictions.shape == test_label_weight_masks.shape
    num_samples = test_labels.shape[0]
    # after np.vstack, both test_labels and test_predictions are arrays of size (num_samples, num_classes)

    # ipdb.set_trace()

    # there can be some trivial samples, but they should be dealt with in a special way
    # since we do not do it here, we assert that there are no trivial samples and ask the user to drop them beforehand
    num_trivial_samples = np.sum(test_label_weight_masks.sum(axis=1) == 0)
    print(
        f"\n \t Trivial samples / Total samples: {num_trivial_samples} / {num_samples} = {num_trivial_samples / num_samples}\n")
    assert num_trivial_samples == 0, f"There are {num_trivial_samples} trivial samples in the {mode} set. Please drop the from the dataset beforehand."

    # ipdb.set_trace()

    class_aucs, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, config.num_classes, pos_label=1,
                                                        mode=mode, test_label_weight_masks=test_label_weight_masks)

    # threshold optimization should only be done on the validation set
    # so multi_label_roc will change depending on the mode
    if mode == 'test' and (saved_thresholds is not None):
        thresholds_optimal = saved_thresholds

    # ipdb.set_trace()

    # Convert thresholds_optimal to a NumPy array and reshape it (reshaping will error )
    thresholds_optimal_array = np.array(thresholds_optimal).reshape(1, config.num_classes)  # (1, num_classes)
    # Binarize the predictions using the optimal thresholds
    threshold_matrix = np.repeat(thresholds_optimal_array, test_predictions.shape[0],
                                 axis=0)  # (num_samples, num_classes)
    # Create a binary mask where values >= thresholds_optimal are set to 1, and others to 0
    test_predictions_binarized = (test_predictions >= threshold_matrix).astype(int)

    # for each class, extract elements with non-zero weight and compute: accuracy, precision, recall, f1 score
    class_accuracies = []
    class_precisions = []
    class_recalls = []
    class_f1_scores = []
    class_num_valid_samples = []
    for c in range(config.num_classes):
        num_valid_samples = np.sum(test_label_weight_masks[:, c])
        class_num_valid_samples.append(num_valid_samples)
        if num_valid_samples > 0:
            # Calculate metrics only if there are valid samples for this class
            accuracy = accuracy_score(y_true=test_labels[:, c], y_pred=test_predictions_binarized[:, c],
                                      sample_weight=test_label_weight_masks[:, c])
            precision = precision_score(y_true=test_labels[:, c], y_pred=test_predictions_binarized[:, c],
                                        sample_weight=test_label_weight_masks[:, c])
            recall = recall_score(y_true=test_labels[:, c], y_pred=test_predictions_binarized[:, c],
                                  sample_weight=test_label_weight_masks[:, c])
            f1 = f1_score(y_true=test_labels[:, c], y_pred=test_predictions_binarized[:, c],
                          sample_weight=test_label_weight_masks[:, c])

            class_accuracies.append(accuracy)
            class_precisions.append(precision)
            class_recalls.append(recall)
            class_f1_scores.append(f1)
        else:
            # If no valid samples, append a None or np.nan to indicate this class was not evaluated
            class_accuracies.append(np.nan)
            class_precisions.append(np.nan)
            class_recalls.append(np.nan)
            class_f1_scores.append(np.nan)

    # ipdb.set_trace()

    masked_predictions = test_predictions_binarized * test_label_weight_masks
    masked_labels = test_labels * test_label_weight_masks

    # Subset Accuracy: the sample is correctly classified if labels for all classes are predicted correctly
    #   https://scikit-learn.org/stable/modules/model_evaluation.html#:~:text=In%20multilabel%20classification%2C%20the%20function,1.0%3B%20otherwise%20it%20is%200.0.
    masked_prediction_fully_correct_statuses = (masked_labels == masked_predictions).all(axis=1)
    avg_score = np.mean(masked_prediction_fully_correct_statuses)

    # ipdb.set_trace()

    logs_dict = {
        f"{mode}_modified_accuracy": avg_score,
        f"{mode}_loss": avg_loss_per_sample,
    }
    for i in range(config.num_classes):
        logs_dict[f"{mode}_class{i}_auc"] = class_aucs[i]
        logs_dict[f"{mode}_class{i}_threshold"] = thresholds_optimal[i]
        logs_dict[f"{mode}_class{i}_accuracy"] = class_accuracies[i]
        logs_dict[f"{mode}_class{i}_precision"] = class_precisions[i]
        logs_dict[f"{mode}_class{i}_recall"] = class_recalls[i]
        logs_dict[f"{mode}_class{i}_f1_score"] = class_f1_scores[i]
        logs_dict[f"{mode}_class{i}_num_valid_samples"] = class_num_valid_samples[i]

    # logging is done in the main function
    # wandb.log(logs_dict)

    return avg_loss_per_sample, avg_score, class_aucs, thresholds_optimal, logs_dict


def main():
    parser = argparse.ArgumentParser(description='Train MIL on patch features.')
    # config path
    parser.add_argument('--base_config_path', type=str, default='./configs/base_config.yaml',
                        help='Path to the base config file')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the current config file. Result will be the base config file with some fields updated by the current config file.')
    args = parser.parse_args()
    # load config
    config = Config.from_yaml(base_path=args.base_config_path, current_path=args.config_path)
    
    # Set seeds for reproducibility
    seed_everything(config)

    # set gpu device
    device_srt_name = f"{config.accelerator}:{config.gpu_device}" \
        if (config.accelerator == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Computations will be performed on {device_srt_name}")
    torch_device = torch.device(device_srt_name)
    # gpu_ids = [config.gpu_device]
    # os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config=config.model_dump(),
        mode="offline",  # faster on compute nodes, `wandb sync <run directory>` on login node
    )
    # config = wandb.config  # TODO: check what it does
    # ipdb.set_trace()

    data_module = LungSubtypingSlideEmbeddingDM(config=config, device=torch_device)
    data_module.setup()
    print("Data module setup complete.\n")

    # model = torch.nn.Linear(
    #     in_features=config.feats_size,
    #     out_features=config.num_classes,
    # )
    model = torch.nn.Sequential(
        torch.nn.Linear(config.feats_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, config.num_classes),
    )
    model = model.to(device=torch_device)
    print(f"Model created and put on device: {torch_device}\n")

    # load weights if a proper path is given as an argument
    if os.path.isfile(config.saved_weights_pth):
        print(f'\nLoading weights from:\n\t{config.saved_weights_pth}\n')
        model.load_state_dict(torch.load(config.saved_weights_pth), strict=True)
        print("Successfully loaded weights.\n")
    else:
        print(f'\nNo weights found at: {config.saved_weights_pth}. Using random initialization of weights.\n')

    if config.task_type == "multi_label_classification":
        criterion = partial(custom_cross_entropy_loss, loss_type='binary_cross_entropy',
                            reduction_after_reweighting='sum')
    elif config.task_type == 'multi_class_classification':
        criterion = partial(custom_cross_entropy_loss, loss_type='cross_entropy',
                            reduction_after_reweighting='sum')
    else:
        raise NotImplementedError("Invalid task_type. Choose 'multi_label_classification' or 'multi_class_classification'.")


    if 'train' in config.train_test_mode:

        optimizer_class = load_object(config.optimizer)
        optimizer = optimizer_class(model.parameters(), **config.optimizer_kwargs)
        
        scheduler = None
        if config.scheduler:
            scheduler_class = load_object(config.scheduler)
            scheduler = scheduler_class(optimizer, **config.scheduler_kwargs)

        train_dataloader = data_module.train_dataloader()
        valid_dataloader = data_module.val_dataloader()

        # ipdb.set_trace()

        best_score = 0
        save_dir_path = os.path.join('weights', datetime.date.today().strftime("%Y-%m-%d"))
        os.makedirs(save_dir_path, exist_ok=True)
        total_runs = len(glob.glob(os.path.join(save_dir_path, '*.pth')))
        save_path = os.path.join(save_dir_path, f'run-{total_runs + 1}-experiment-{config.experiment_name}.pth')

        thresholds_dir_path = save_dir_path.replace('weights', 'thresholds')
        os.makedirs(thresholds_dir_path, exist_ok=True)
        thresholds_file_path = save_path.replace('weights', 'thresholds').replace('.pth', '.json')

        # save an empty file to indicate that the run has started
        open(save_path, 'a').close()
        print(f"Created an empty file for saving weights to indicate that the run has started:\n\t{save_path}\n")

        for epoch in range(1, config.num_epochs + 1):
            print(f"Epoch [{epoch}/{config.num_epochs}]")
            train_loss_bag = train(train_dataloader=train_dataloader, model=model, criterion=criterion,
                                   optimizer=optimizer, config=config, device=torch_device)
            valid_loss_bag, avg_score, aucs, thresholds_optimal, valid_logs_dict = evaluate(
                eval_dataloader=valid_dataloader, model=model, criterion=criterion, mode='valid',
                saved_thresholds=None, config=config, device=torch_device)
            current_score = (sum(aucs) + avg_score) / (len(aucs) + 1)

            valid_logs_dict["train_loss"] = train_loss_bag
            valid_logs_dict['previous_best_score'] = best_score
            valid_logs_dict['current_score'] = current_score
            wandb.log(valid_logs_dict)

            print(
                '\r \t Train loss: %.4f Valid loss: %.4f, Proportion of samples with fully correct labels: %.4f, AUC: ' %
                (train_loss_bag, valid_loss_bag, avg_score) + '|'.join(
                    'class-{}>>{:.4f}'.format(*k) for k in enumerate(aucs)))
            if scheduler is not None:
                scheduler.step()
            if current_score >= best_score:
                best_score = current_score
                torch.save(model.state_dict(), save_path)
                print('\t Best model saved at: ' + save_path)
                print('\t Best thresholds ===>>> ' + '|'.join(
                    'class-{}>>{:.4f}'.format(*k) for k in enumerate(thresholds_optimal)))

                # ipdb.set_trace()

                # if records_thresholds.json does not exist, create it and save the thresholds converting them to a list of strings
                # if it exists, load it, update it, and save it again
                if not os.path.exists(thresholds_file_path):
                    with open(thresholds_file_path, 'w') as f:
                        json.dump({save_path: [str(threshold) for threshold in thresholds_optimal]}, f)
                else:
                    with open(thresholds_file_path, 'r') as f:
                        records_thresholds = json.load(f)
                    records_thresholds[save_path] = [str(threshold) for threshold in thresholds_optimal]
                    with open(thresholds_file_path, 'w') as f:
                        json.dump(records_thresholds, f)

    if 'test' in config.train_test_mode:

        # if trained before, use the best model
        if 'train' in config.train_test_mode:
            print(f'\nTesting the model from weights saved during training:\n\t{save_path}\n')
            model.load_state_dict(torch.load(save_path))
            saved_thresholds_name = save_path
        # if not trained before, use the initial weights - they have been loaded prior to entering
        #   if 'train' in config.train_test_mode:
        else:
            print(f'\nTesting the model from weights path given as an argument:\n\t{config.saved_weights_pth}\n')
            saved_thresholds_name = config.saved_weights_pth

        # load the thresholds from records_thresholds.json
        #   `thresholds_file_path` is always derived from `saved_thresholds_name`
        thresholds_file_path = saved_thresholds_name.replace('weights', 'thresholds').replace('.pth', '.json')
        with open(thresholds_file_path, 'r') as f:
            records_thresholds = json.load(f)

        saved_thresholds = [float(threshold) for threshold in records_thresholds[saved_thresholds_name]]
        print(f'Using thresholds from  ===>>> {saved_thresholds_name} ' + '|'.join(
            'class-{}>>{:.4f}'.format(*k) for k in enumerate(saved_thresholds)))

        test_dataloader = data_module.test_dataloader()
        test_loss_bag, avg_score, aucs, _, test_logs_dict = evaluate(eval_dataloader=test_dataloader, model=model,
                                                                     criterion=criterion, mode='test',
                                                                     saved_thresholds=saved_thresholds, config=config,
                                                                     device=torch_device)
        wandb.log(test_logs_dict)
        print('\r Test loss: %.4f, Proportion of samples with fully correct labels: %.4f, AUC: ' %
              (test_loss_bag, avg_score) + '|'.join('class-{}>>{:.4f}'.format(*k) for k in enumerate(aucs)))


if __name__ == '__main__':
    main()
