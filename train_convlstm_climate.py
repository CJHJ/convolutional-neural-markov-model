'''Train or test deep learning model'''
import logging
from tqdm import tqdm

import numpy as np
from pathlib import Path
import argparse
import random
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data import CMAPDataset, get_netcdf_data, normalize, denormalize
from utils.log import log_evaluation
from models.convlstm import ConvLSTM


def main(args):
    # Init tensorboard
    writer = SummaryWriter()
    model_name = 'ConvLSTM'
    training_log_loc = Path(
        './logs/{}/training.log'.format(model_name.lower()))

    # Set evaluation log file
    evaluation_logpath = './logs/{}/evaluation_climate_result.log'.format(
        model_name.lower())
    log_evaluation(evaluation_logpath,
                   'Evaluation Trial - {}\n'.format(args.trialnumber))

    # Constants
    time_length = 30
    input_length = 20
    pred_length = time_length - input_length
    validation_pred_lengths = [20]
    train_batch_size = 16
    valid_batch_size = 1
    training_size_ratio = 0.7
    data_min_val, data_max_val = 0, 80
    log_dir = './logs/convlstm/training.log'

    # Device checking
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Make dataset
    logging.info("Generate data")
    temp_train_data, temp_valid_data = get_netcdf_data(
        args.datapath, training_size_ratio, time_length)

    # Replace valid data with pkl file
    logging.info("Train data shape: {}", temp_train_data.shape)
    logging.info("Valid data shape: {}", temp_valid_data.shape)

    train_dataset = CMAPDataset(torch.Tensor(temp_train_data))
    valid_dataset = CMAPDataset(torch.Tensor(temp_valid_data))

    # Create data loaders from pickle data
    logging.info("Generate data loaders")
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, num_workers=4, shuffle=False)

    # Training parameters
    row_dim = temp_train_data.shape[3]
    col_dim = temp_train_data.shape[2]
    input_dim = row_dim * col_dim

    # Create model
    logging.warning("Generate model")
    logging.warning(input_dim)
    input_channels = 1
    hidden_channels = [64]
    kernel_size = 3
    pred_input_dim = 10
    model = ConvLSTM(input_channels, hidden_channels,
                     kernel_size, pred_input_dim, device)

    # Initialize model
    logging.info("Initialize model")
    epochs = args.endepoch
    learning_rate = 0.001

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Starting epoch
    start_epoch = args.startepoch

    # Load model
    if start_epoch != 0:
        load_model = Path('./checkpoints/ConvLSTMClimate/e{}-i119-tn{}.pt'.format(
            start_epoch - 1, args.trialnumber)
        )
        if load_model is not None:
            logging.info("loading model from %s..." % load_model)
            model.load_state_dict(torch.load(load_model, map_location=device))

    # Validation only?
    validation_only = args.validonly

    # Train the model
    if not validation_only:
        logging.info("Training model")
        train_losses = []
        valid_losses = []
        for epoch in tqdm(range(start_epoch, epochs), desc='Epoch', leave=True):
            r_loss_train = 0
            model.train(True)
            idx = 0
            mov_avg_loss = 0
            for data in tqdm(train_dataloader, desc='Train', leave=True):
                optimizer.zero_grad()

                data['observation'] = normalize(
                    data['observation'].unsqueeze(2).to(device),
                    data_min_val, data_max_val)
                batch_size, length, _, w, h = data['observation'].shape

                preds = model(data['observation']
                              [:, :-pred_length, :, :, :], pred_length)

                loss = criterion(
                    preds[0], data['observation'][:, input_length:, :, :, :])
                loss.backward()
                optimizer.step()

                # Running losses
                mov_avg_loss += loss.item() * data['observation'].size(0)

                r_loss_train += loss.item() * data['observation'].size(0)
                idx += 1

            # Average losses
            train_loss_avg = r_loss_train / len(train_dataset)
            writer.add_scalar('Loss/train', train_loss_avg, epoch)
            logging.info("Epoch: %d, Training loss: %1.5f",
                         epoch, train_loss_avg)

            # # Time to time evaluation
            if epoch == epochs - 1:
                # Save model
                torch.save(model.state_dict(), args.modelsavepath /
                           'e{}-i{}-tn{}.pt'.format(epoch, idx, args.trialnumber))

                for temp_pred_length in validation_pred_lengths:
                    r_loss_valid = 0
                    r_loss_loc_valid = 0
                    model.train(False)
                    model.eval()
                    val_input_length = time_length - temp_pred_length
                    with torch.no_grad():
                        for i, data in enumerate(tqdm(valid_dataloader, desc='Valid', leave=True)):
                            data['observation'] = normalize(
                                data['observation'].unsqueeze(2).to(device),
                                data_min_val, data_max_val)
                            batch_size, length, _, w, h = data['observation'].shape

                            preds = model.forward_with_gt(
                                data['observation'], temp_pred_length)

                            loss = criterion(
                                preds, data['observation'][:, val_input_length:, :, :, :])

                            ground_truth = denormalize(
                                data['observation'].squeeze().cpu(
                                ).detach(), data_min_val, data_max_val
                            )
                            pred_with_input = denormalize(
                                torch.cat(
                                    [data['observation'][:, :-temp_pred_length, :, :, :].squeeze(),
                                     preds.squeeze()], dim=0
                                ).cpu().detach(), data_min_val, data_max_val
                            )

                            # Running losses
                            r_loss_valid += loss.item() * \
                                data['observation'].size(0)
                            r_loss_loc_valid += np.sum((preds.detach().cpu().numpy(
                            ) - data['observation'][:, val_input_length:, :, :, :].detach().cpu().numpy()) ** 2)

                    # Average losses
                    valid_loss_avg = r_loss_valid / len(valid_dataset)
                    valid_loss_loc_avg = r_loss_loc_valid / \
                        (len(valid_dataset) * temp_pred_length * row_dim * col_dim)
                    writer.add_scalar(
                        'Loss/test_obs', valid_loss_loc_avg, epoch)
                    writer.add_scalar('Loss/test', valid_loss_avg, epoch)
                    logging.info("Validation loss: %1.5f", valid_loss_avg)
                    logging.info("Validation obs loss: %1.5f",
                                 valid_loss_loc_avg)
                    log_evaluation(evaluation_logpath, "Validation obs loss for {}s pred ConvLSTM: {}\n".format(
                        temp_pred_length, valid_loss_loc_avg))

            # Save model
            if args.savemodel:
                torch.save(model.state_dict(), args.modelsavepath /
                           'e{}-i{}-tn{}.pt'.format(epoch, idx, args.trialnumber))

    # Last validation after training
    test_samples_indices = range(temp_valid_data.shape[0])
    total_n = 0
    val_pred_length = args.validpredlength
    val_input_length = time_length - val_pred_length
    if validation_only:
        r_loss_valid = 0
        r_loss_loc_valid = 0
        r_loss_latent_valid = 0
        model.train(False)
        with torch.no_grad():
            for i in tqdm(test_samples_indices, desc='Valid', leave=True):
                # Data processing
                data = valid_dataset[i]
                if torch.isnan(torch.sum(data['observation'])):
                    print("Skip {}".format(i))
                    continue
                else:
                    total_n += 1
                data['observation'] = normalize(
                    data['observation'].unsqueeze(0).unsqueeze(2).to(device),
                    data_min_val, data_max_val)
                batch_size, length, _, w, h = data['observation'].shape

                preds = model.forward_with_gt(
                    data['observation'], val_pred_length)

                loss = criterion(
                    preds, data['observation'][:, val_input_length:, :, :, :])

                ground_truth = denormalize(
                    data['observation'].squeeze().cpu().detach(),
                    data_min_val, data_max_val
                )
                pred_with_input = denormalize(
                    torch.cat(
                        [data['observation'][:, :-val_pred_length, :, :, :].squeeze(),
                         preds.squeeze()], dim=0
                    ).cpu().detach(),
                    data_min_val, data_max_val
                )

                # Save samples
                if i < 5:
                    save_dir_samples = Path(
                        './samples/climate/{}s'.format(val_pred_length))
                    # with open(save_dir_samples / '{}-gt.pkl'.format(i), 'wb') as fout:
                    #     pickle.dump(ground_truth, fout)
                    with open(save_dir_samples / '{}-convlstm-pred.pkl'.format(i), 'wb') as fout:
                        pickle.dump(pred_with_input, fout)

                # Running losses
                r_loss_valid += loss.item() * data['observation'].size(0)
                r_loss_loc_valid += np.sum((preds.detach().cpu().numpy(
                ) - data['observation'][:, val_input_length:, :, :, :].detach().cpu().numpy()) ** 2)
                r_loss_latent_valid += np.sum((preds.squeeze().detach().cpu().numpy(
                ) - data['latent'][val_input_length:, :, :].detach().cpu().numpy()) ** 2)

        # Average losses
        print(total_n)

        valid_loss_avg = r_loss_valid / total_n
        valid_loss_loc_avg = r_loss_loc_valid / \
            (total_n * val_pred_length * col_dim * row_dim)
        valid_loss_latent_avg = r_loss_latent_valid / \
            (total_n * val_pred_length * col_dim * row_dim)
        logging.info("Validation loss for %ds pred ConvLSTM: %f",
                     val_pred_length, valid_loss_avg)
        logging.info("Validation obs loss: %f", valid_loss_loc_avg)
        logging.info("Validation latent loss: %f", valid_loss_latent_avg)

        with open('ConvLSTMResult.log', 'a+') as fout:
            validation_log = 'Pred {}s ConvLSTM: {}\n'.format(
                val_pred_length, valid_loss_loc_avg)
            fout.write(validation_log)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(logging.getLogger().getEffectiveLevel())

    parser = argparse.ArgumentParser(description="Train a dnn model")
    parser.add_argument('--datapath', '-dp',
                        default='./data/cmap/precip.pentad.mean.nc', type=str, help="Data path")
    parser.add_argument('--modelsavepath', '-msp',
                        default='./checkpoints/ConvLSTMClimate', type=str, help="Model path")
    parser.add_argument('--validpredlength', '-vpl', default=5,
                        type=int, help="Validation prediction length")
    parser.add_argument('--validonly', '-vo', default=False, type=bool,
                        help="Go to training mode if false, validation mode if true")
    parser.add_argument('--savemodel', '-sm', default=True, type=bool,
                        help="Save model if true, don't if false")
    parser.add_argument('--trialnumber', '-tn', default=0, type=int,
                        help="Set training trial number")
    parser.add_argument('--startepoch', '-se', default=0,
                        type=int, help="Set starting epoch")
    parser.add_argument('--endepoch', '-ee', default=299,
                        type=int, help="Set ending epoch")

    args = parser.parse_args()

    args.datapath = Path(args.datapath)
    args.modelsavepath = Path(args.modelsavepath)

    main(args)
