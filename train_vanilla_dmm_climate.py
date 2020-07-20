'''Train or test deep learning model'''
import logging
from tqdm import tqdm

import numpy as np
from pathlib import Path
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data import CMAPDataset, get_netcdf_data, normalize, denormalize
from utils.log import log_evaluation
from models.dmm import DMM, reverse_sequences, do_prediction, do_prediction_rep_inference

from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO


def main(args):
    # Init tensorboard
    writer = SummaryWriter('./runs/' + args.runname + str(args.trialnumber))
    model_name = 'VanillaDMMClimate'
    training_log_loc = Path(
        './logs/{}/training.log'.format(model_name.lower()))

    # Set evaluation log file
    evaluation_logpath = './logs/{}/evaluation_climate_result.log'.format(
        model_name.lower())
    log_evaluation(evaluation_logpath,
                   'Evaluation Trial - {}\n'.format(args.trialnumber))

    # Constants
    time_length = 30
    input_length_for_pred = 20
    pred_length = time_length - input_length_for_pred
    validation_pred_lengths = [5, 10, 15, 20]
    train_batch_size = 16
    valid_batch_size = 1
    training_size_ratio = 0.7
    data_min_val, data_max_val = 0, 80

    # For model
    input_channels = 1
    z_channels = 50
    emission_channels = [64, 32]
    transition_channels = 64
    encoder_channels = [32, 64]
    rnn_input_dim = 256
    rnn_channels = 128
    kernel_size = 3
    pred_length = 10

    # Device checking
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Make dataset
    logging.info("Generate data")
    temp_train_data, temp_valid_data = get_netcdf_data(
        args.datapath, training_size_ratio, time_length)
    logging.info("Train data shape: {}".format(temp_train_data.shape))
    logging.info("Valid data shape: {}".format(temp_valid_data.shape))
    train_dataset = CMAPDataset(torch.Tensor(temp_train_data))
    valid_dataset = CMAPDataset(torch.Tensor(temp_valid_data))

    # Create data loaders from pickle data
    logging.info("Generate data loaders")
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, num_workers=4)

    # Training parameters
    width = temp_train_data.shape[3]
    height = temp_train_data.shape[2]
    input_dim = width * height

    # Create model
    logging.warning("Generate model")
    logging.warning("Height and width: {}, {}".format(height, width))
    logging.warning(input_dim)
    pred_input_dim = 10
    dmm = DMM(input_channels=input_channels, z_channels=z_channels, emission_channels=emission_channels,
              transition_channels=transition_channels, encoder_channels=encoder_channels, rnn_input_dim=rnn_input_dim, rnn_channels=rnn_channels, kernel_size=kernel_size, height=height, width=width, pred_input_dim=pred_input_dim, num_layers=1, rnn_dropout_rate=0.0,
              num_iafs=0, iaf_dim=50, use_cuda=use_cuda)

    # Initialize model
    logging.info("Initialize model")
    epochs = args.endepoch
    learning_rate = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    clip_norm = 10.0
    lr_decay = 1.0
    weight_decay = 0
    adam_params = {"lr": learning_rate, "betas": (beta1, beta2),
                   "clip_norm": clip_norm, "lrd": lr_decay,
                   "weight_decay": weight_decay}
    adam = ClippedAdam(adam_params)
    elbo = Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # saves the model and optimizer states to disk
    save_model = Path('./checkpoints/' + model_name)

    def save_checkpoint(epoch):
        save_dir = save_model / '{}.model'.format(epoch)
        save_opt_dir = save_model / '{}.opt'.format(epoch)
        logging.info("saving model to %s..." % save_dir)
        torch.save(dmm.state_dict(), save_dir)
        logging.info("saving optimizer states to %s..." % save_opt_dir)
        adam.save(save_opt_dir)
        logging.info("done saving model and optimizer checkpoints to disk.")

    # Staring epoch
    start_epoch = args.startepoch

    # loads the model and optimizer states from disk
    if start_epoch != 0:
        load_opt = './checkpoints/' + model_name + \
            '/e{}-i119-tn{}-opt.opt'.format(start_epoch - 1, args.trialnumber)
        load_model = './checkpoints/' + model_name + \
            '/e{}-i119-tn{}.pt'.format(start_epoch - 1, args.trialnumber)

        def load_checkpoint():
            # assert exists(load_opt) and exists(load_model), \
            #     "--load-model and/or --load-opt misspecified"
            logging.info("loading model from %s..." % load_model)
            dmm.load_state_dict(torch.load(load_model, map_location=device))
            # logging.info("loading optimizer states from %s..." % load_opt)
            # adam.load(load_opt)
            # logging.info("done loading model and optimizer states.")

        if load_model != '':
            logging.info('Load checkpoint')
            load_checkpoint()

    # Validation only?
    validation_only = args.validonly

    # Train the model
    if not validation_only:
        logging.info("Training model")
        annealing_epochs = 1000
        minimum_annealing_factor = 0.2
        N_train_size = temp_train_data.shape[0]
        N_mini_batches = int(N_train_size / train_batch_size +
                             int(N_train_size % train_batch_size > 0))
        for epoch in tqdm(range(start_epoch, epochs), desc='Epoch', leave=True):
            r_loss_train = 0
            dmm.train(True)
            idx = 0
            mov_avg_loss = 0
            mov_data_len = 0
            for which_mini_batch, data in enumerate(tqdm(train_dataloader, desc='Train', leave=True)):
                if annealing_epochs > 0 and epoch < annealing_epochs:
                    # compute the KL annealing factor approriate for the current mini-batch in the current epoch
                    min_af = minimum_annealing_factor
                    annealing_factor = min_af + (1.0 - min_af) * \
                        (float(which_mini_batch + epoch * N_mini_batches + 1) /
                         float(annealing_epochs * N_mini_batches))
                else:
                    # by default the KL annealing factor is unity
                    annealing_factor = 1.0

                data['observation'] = normalize(
                    data['observation'].unsqueeze(2).to(device), data_min_val, data_max_val)
                batch_size, length, _, w, h = data['observation'].shape
                data_reversed = reverse_sequences(data['observation'])
                data_mask = torch.ones(
                    batch_size, length, input_channels, w, h).cuda()

                loss = svi.step(data['observation'],
                                data_reversed, data_mask, annealing_factor)

                # Running losses
                mov_avg_loss += loss
                mov_data_len += batch_size

                r_loss_train += loss
                idx += 1

            # Average losses
            train_loss_avg = r_loss_train / (len(train_dataset) * time_length)
            writer.add_scalar('Loss/train', train_loss_avg, epoch)
            logging.info("Epoch: %d, Training loss: %1.5f",
                         epoch, train_loss_avg)

            # # Time to time evaluation
            if epoch == epochs - 1:
                for temp_pred_length in validation_pred_lengths:
                    r_loss_valid = 0
                    r_loss_loc_valid = 0
                    r_loss_scale_valid = 0
                    dmm.train(False)
                    val_pred_length = temp_pred_length
                    val_pred_input_length = 10
                    with torch.no_grad():
                        for i, data in enumerate(tqdm(valid_dataloader, desc='Eval', leave=True)):
                            data['observation'] = normalize(
                                data['observation'].unsqueeze(2).to(device), data_min_val, data_max_val)
                            batch_size, length, _, w, h = data['observation'].shape
                            data_reversed = reverse_sequences(
                                data['observation'])
                            data_mask = torch.ones(
                                batch_size, length, input_channels, w, h).cuda()

                            pred_tensor = data['observation'][:,
                                                              :input_length_for_pred, :, :, :]
                            pred_tensor_reversed = reverse_sequences(
                                pred_tensor)
                            pred_tensor_mask = torch.ones(
                                batch_size, input_length_for_pred, input_channels, w, h).cuda()

                            ground_truth = data['observation'][:,
                                                               input_length_for_pred:, :, :, :]

                            val_nll = svi.evaluate_loss(
                                data['observation'], data_reversed, data_mask)
                            # _, _, loss_loc, loss_scale = do_prediction(
                            #     dmm, pred_tensor, pred_tensor_reversed, pred_tensor_mask, val_pred_length, ground_truth)
                            preds, _, loss_loc, loss_scale = do_prediction_rep_inference(
                                dmm, pred_tensor_mask, val_pred_length, val_pred_input_length, data['observation'])

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

                            # Running losses
                            r_loss_valid += val_nll
                            r_loss_loc_valid += loss_loc
                            r_loss_scale_valid += loss_scale

                        # Average losses
                        valid_loss_avg = r_loss_valid / \
                            (len(valid_dataset) * time_length)
                        valid_loss_loc_avg = r_loss_loc_valid / \
                            (len(valid_dataset) * val_pred_length * width * height)
                        valid_loss_scale_avg = r_loss_scale_valid / \
                            (len(valid_dataset) * val_pred_length * width * height)
                        writer.add_scalar('Loss/test', valid_loss_avg, epoch)
                        writer.add_scalar(
                            'Loss/test_obs', valid_loss_loc_avg, epoch)
                        writer.add_scalar('Loss/test_scale',
                                          valid_loss_scale_avg, epoch)
                        logging.info("Validation loss: %1.5f", valid_loss_avg)
                        logging.info("Validation obs loss: %1.5f",
                                     valid_loss_loc_avg)
                        logging.info("Validation scale loss: %1.5f",
                                     valid_loss_scale_avg)
                        log_evaluation(evaluation_logpath, "Validation obs loss for {}s pred {}: {}\n".format(
                            val_pred_length, args.trialnumber, valid_loss_loc_avg))
                        log_evaluation(evaluation_logpath, "Validation scale loss for {}s pred {}: {}\n".format(
                            val_pred_length, args.trialnumber, valid_loss_scale_avg))

                # Save model
                torch.save(dmm.state_dict(), args.modelsavepath / model_name /
                           'e{}-i{}-tn{}.pt'.format(epoch, idx, args.trialnumber))
                adam.save(args.modelsavepath / model_name /
                          'e{}-i{}-tn{}-opt.opt'.format(epoch, idx, args.trialnumber))

    # Last validation after training
    test_samples_indices = range(temp_valid_data.shape[0])
    total_n = 0
    if validation_only:
        r_loss_loc_valid = 0
        r_loss_scale_valid = 0
        r_loss_latent_valid = 0
        dmm.train(False)
        val_pred_length = args.validpredlength
        val_pred_input_length = 10
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
                    data['observation'].unsqueeze(0).unsqueeze(2).to(device), data_min_val, data_max_val)
                batch_size, length, _, w, h = data['observation'].shape
                data_reversed = reverse_sequences(data['observation'])
                data_mask = torch.ones(
                    batch_size, length, input_channels, w, h).to(device)

                # Prediction
                pred_tensor_mask = torch.ones(
                    batch_size, input_length_for_pred, input_channels, w, h).to(device)

                preds, _, loss_loc, loss_scale = do_prediction_rep_inference(
                    dmm, pred_tensor_mask, val_pred_length, val_pred_input_length, data['observation'])

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
                    with open(save_dir_samples / '{}-gt.pkl'.format(i), 'wb') as fout:
                        pickle.dump(ground_truth, fout)
                    with open(save_dir_samples / '{}-vanilladmm-pred.pkl'.format(i), 'wb') as fout:
                        pickle.dump(pred_with_input, fout)

                # Running losses
                r_loss_loc_valid += loss_loc
                r_loss_scale_valid += loss_scale
                r_loss_latent_valid += np.sum((preds.squeeze().detach().cpu().numpy(
                ) - data['latent'][time_length - val_pred_length:, :, :].detach().cpu().numpy()) ** 2)

        # Average losses
        test_samples_indices = range(total_n)
        print(total_n)
        valid_loss_loc_avg = r_loss_loc_valid / \
            (total_n * val_pred_length * width * height)
        valid_loss_scale_avg = r_loss_scale_valid / \
            (total_n * val_pred_length * width * height)
        valid_loss_latent_avg = r_loss_latent_valid / \
            (total_n * val_pred_length * width * height)
        logging.info("Validation obs loss for %ds pred VanillaDMM: %f",
                     val_pred_length, valid_loss_loc_avg)
        logging.info("Validation latent loss: %f", valid_loss_latent_avg)

        with open('VanillaDMMClimateResult.log', 'a+') as fout:
            validation_log = 'Pred {}s VanillaDMM: {}\n'.format(
                val_pred_length, valid_loss_loc_avg)
            fout.write(validation_log)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(logging.getLogger().getEffectiveLevel())

    parser = argparse.ArgumentParser(description="Train a dnn model")
    parser.add_argument('--datapath', '-dp',
                        default='./data/cmap/precip.pentad.mean.nc', type=str, help="Data path")
    parser.add_argument('--modelsavepath', '-msp',
                        default='./checkpoints', type=str, help="Model path")
    parser.add_argument('--validpredlength', '-vpl', default=5,
                        type=int, help="Validation prediction length")
    parser.add_argument('--validonly', '-vo', default=False, type=bool,
                        help="Go to training mode if false, validation mode if true")
    parser.add_argument('--trialnumber', '-tn', default=0, type=int,
                        help="Set training trial number")
    parser.add_argument('--startepoch', '-se', default=0,
                        type=int, help="Set starting epoch")
    parser.add_argument('--endepoch', '-ee', default=299,
                        type=int, help="Set ending epoch")
    parser.add_argument('--runname', '-rn', default="VanillaDMM-Climate",
                        type=str, help="Set training record name")

    args = parser.parse_args()

    args.datapath = Path(args.datapath)
    args.modelsavepath = Path(args.modelsavepath)

    main(args)
