"""
Taken from pyro tutorial
"""

import argparse
from os.path import exists

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from tqdm import tqdm


class Emitter(nn.Module):
    '''
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`


    Args:
        width (int): Width of input image
        height (int): Height of input image
        input_channels (int): Channel size of input image
        z_dim (int): Dimensional size of latent variable
        emission_channels (tuple of int): Channel size of hidden layers
        kernel_size (int): Kernel size of layers
    '''

    def __init__(self, width, height, input_channels, z_dim, emission_channels, kernel_size):
        super().__init__()
        # Parameter initialization
        # TODO: Deconv kernel size should be even
        kernel_size = 4
        padding = int((kernel_size - 1) / 2)
        n_layers = 2
        stride = 2
        self.feature_to_cnn_dim = min(width, height) // 2 ** n_layers
        self.feature_to_cnn_shape = (
            emission_channels[0], self.feature_to_cnn_dim, self.feature_to_cnn_dim)

        # Initialize the three conv transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(
            z_dim, np.prod(self.feature_to_cnn_shape))
        self.lin_hidden_to_hidden = nn.ConvTranspose2d(
            emission_channels[0], emission_channels[1], kernel_size, stride, padding, bias=True)
        self.lin_hidden_to_input_loc = nn.ConvTranspose2d(
            emission_channels[1], input_channels, kernel_size, stride, padding, bias=True)
        self.lin_hidden_to_input_scale = nn.ConvTranspose2d(
            emission_channels[1], input_channels, kernel_size, stride, padding, bias=True)

        # Initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        batch_size = z_t.shape[0]
        h1 = self.relu(self.lin_z_to_hidden(z_t)).view(batch_size,
                                                       *self.feature_to_cnn_shape)
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        loc = self.tanh(self.lin_hidden_to_input_loc(h2))
        scale = self.softplus(
            self.lin_hidden_to_input_scale(h2)).clamp(min=1e-4)
        return loc, scale


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()

        # initialize the six conv transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))

        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)

        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean

        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(
            self.relu(proposed_mean))).clamp(min=1e-4)

        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)

        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)

        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))

        # return loc, scale which can be fed into Normal
        return loc, scale


class ObservationEncoder(nn.Module):
    '''
    Encoder for encoding 2d data to be feed as RNN's input


    Args:

    '''

    def __init__(self, width, height, input_channels, rnn_dim, encoder_channels, kernel_size):
        super().__init__()
        # Parameter initialization
        padding = int((kernel_size - 1) / 2)
        n_layers = 2
        stride = 2
        self.feature_width = width // 2 ** len(encoder_channels)
        self.feature_height = height // 2 ** len(encoder_channels)
        self.feature_dim = encoder_channels[-1] * \
            self.feature_width * self.feature_height

        # Initialize the three conv transformations used in the neural network
        self.conv_x_to_hidden = nn.Conv2d(
            input_channels, encoder_channels[0], kernel_size, stride, padding, bias=True)
        self.conv_hidden_to_hidden = nn.Conv2d(
            encoder_channels[0], encoder_channels[1], kernel_size, stride, padding, bias=True)
        self.lin_hidden_to_rnn = nn.Linear(
            self.feature_dim, rnn_dim)

        # Initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        batch_size = z_t.shape[0]
        h1 = self.relu(self.conv_x_to_hidden(z_t))
        h2 = self.relu(self.conv_hidden_to_hidden(h1)).view(batch_size, -1)
        rnn_input = self.tanh(self.lin_hidden_to_rnn(h2))

        return rnn_input


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, input_channels=1, z_channels=16, emission_channels=[32, 16],
                 transition_channels=32, encoder_channels=[16, 32], rnn_input_dim=32, rnn_channels=32, kernel_size=3, height=100, width=100, pred_input_dim=5, num_layers=1, rnn_dropout_rate=0.0,
                 num_iafs=0, iaf_dim=50, use_cuda=False):
        super().__init__()
        self.input_channels = input_channels
        self.rnn_input_dim = rnn_input_dim

        print("DMM: {}, {}".format(width, height))

        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(
            width, height, input_channels, z_channels, emission_channels, kernel_size)
        self.trans = GatedTransition(z_channels, transition_channels)
        self.combiner = Combiner(z_channels, rnn_channels)
        self.obs_encoder = ObservationEncoder(
            width, height, input_channels, rnn_input_dim, encoder_channels, kernel_size)

        # Instantiate RNN
        if use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_channels,
                           batch_first=True, bidirectional=False, num_layers=num_layers,
                           dropout=rnn_dropout_rate)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_channels))
        self.z_q_0 = nn.Parameter(torch.zeros(z_channels))

        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_channels))
        self.c_0 = nn.Parameter(torch.zeros(1, 1, rnn_channels))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch), dim=-3):
            # sample the latents z and observed x's one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                      dist.Normal(z_loc, z_scale).to_event(1))

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_loc_t, emission_scale_t = self.emitter(z_t)

                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                pyro.sample("obs_x_%d" % t,
                            dist.Normal(emission_loc_t, emission_scale_t)
                            .to_event(1),
                            obs=mini_batch[:, t - 1, :, :, :])

                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask, annealing_factor=1.0):
        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        h_0 = self.h_0.expand(1, mini_batch.size(
            0), self.rnn.hidden_size).contiguous()
        c_0 = self.c_0.expand(1, mini_batch.size(
            0), self.rnn.hidden_size).contiguous()

        # encode every observed x
        batch_size = mini_batch_reversed.shape[0]
        seq_len = mini_batch_reversed.shape[1]
        enc_mini_batch_reversed = torch.zeros(
            batch_size, seq_len, self.rnn_input_dim).to(self.device)
        for t in range(seq_len):
            enc_mini_batch_reversed[:, t, :] = self.obs_encoder(
                mini_batch_reversed[:, t, :, :, :])

        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(enc_mini_batch_reversed, (h_0, c_0))

        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = reverse_sequences(rnn_output)

        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(
                    z_prev, rnn_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                z_dist = dist.Normal(z_loc, z_scale)

                assert z_dist.event_shape == ()
                assert z_dist.batch_shape == (
                    len(mini_batch), self.z_q_0.size(0))

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                    z_t = pyro.sample("z_%d" % t,
                                      z_dist.to_event(1))

                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t

        return z_t


def reverse_sequences(rnn_output):
    T = rnn_output.size(1)
    time_slices = torch.arange(T - 1, -1, -1, device=rnn_output.device)

    return rnn_output.index_select(1, time_slices)


def do_prediction(dmm, pred_batch, pred_batch_reversed, pred_batch_mask, pred_length, ground_truth):
    # Do prediction from previous observations
    with torch.no_grad():
        # Initialization
        bsize, input_seq_len, input_channels, width, height = pred_batch.shape
        # pred_latents_loc = torch.Tensor(bsize, pred_length, input_channels, width, height)
        # pred_latents_scale = torch.Tensor(bsize, pred_length, input_channels, width, height)
        pred_observations_loc = torch.Tensor(
            bsize, pred_length, input_channels, width, height).to(pred_batch.device)
        pred_observations_scale = torch.Tensor(
            bsize, pred_length, input_channels, width, height).to(pred_batch.device)

        # Use guide to calculate latents from observations
        z_prev = dmm.guide(pred_batch, pred_batch_reversed, pred_batch_mask)

        # Use model to predict next latents and generate observation from them
        for i in range(0, pred_length):
            z_pred_loc, z_pred_scale = dmm.trans(z_prev)
            z_t = pyro.sample("z_pred_%d" % i, dist.Normal(
                z_pred_loc, z_pred_scale).to_event(1))

            emission_loc_t, emission_scale_t = dmm.emitter(z_t)
            x_t = pyro.sample("x_pred_%d" % i,
                              dist.Normal(emission_loc_t, emission_scale_t)
                              .to_event(1))

            # Insert into tensors
            # pred_latents_loc[:, i, :, :, :] = z_pred_loc
            # pred_latents_scale[:, i, :, :, :] = z_pred_scale
            pred_observations_loc[:, i, :, :, :] = emission_loc_t
            pred_observations_scale[:, i, :, :, :] = emission_scale_t

            z_prev = z_t

        observations_loss_loc = torch.sum(
            (pred_observations_loc - ground_truth) ** 2).detach().cpu().numpy()
        observations_loss_scale = torch.sum((pred_observations_scale - torch.ones(
            ground_truth.shape).to(pred_batch.device)) ** 2).detach().cpu().numpy()

    return pred_observations_loc, pred_observations_scale, observations_loss_loc, observations_loss_scale


def do_prediction_rep_inference(dmm, pred_batch_mask, pred_length, input_pred_length, ground_truth, summed=True):
    # Do prediction from previous observations
    with torch.no_grad():
        # Initialization
        bsize, input_seq_len, input_channels, width, height = ground_truth.shape
        # pred_latents_loc = torch.Tensor(bsize, pred_length, input_channels, width, height)
        # pred_latents_scale = torch.Tensor(bsize, pred_length, input_channels, width, height)
        pred_observations_loc = torch.Tensor(
            bsize, pred_length, input_channels, width, height).to(ground_truth.device)
        pred_observations_scale = torch.Tensor(
            bsize, pred_length, input_channels, width, height).to(ground_truth.device)

        for i in tqdm(range(pred_length)):
            input_pred_start = input_seq_len - input_pred_length - pred_length + i
            if input_pred_start < 0:
                input_pred_start = 0
            input_pred_end = input_pred_start + input_pred_length
            if input_pred_end > input_seq_len:
                input_pred_end = input_seq_len
            pred_batch = ground_truth[:,
                                      input_pred_start:input_pred_end, :, :, :]
            pred_batch_reversed = reverse_sequences(pred_batch)
            assert pred_batch.shape[1] == input_pred_length

            # Use guide to calculate latents from observations
            z_prev = dmm.guide(
                pred_batch, pred_batch_reversed, pred_batch_mask)

            # Use model to predict next latents and generate observation from them
            z_pred_loc, z_pred_scale = dmm.trans(z_prev)
            z_t = pyro.sample("z_pred_%d" % i, dist.Normal(
                z_pred_loc, z_pred_scale).to_event(1))

            emission_loc_t, emission_scale_t = dmm.emitter(z_t)
            x_t = pyro.sample("x_pred_%d" % i,
                              dist.Normal(emission_loc_t, emission_scale_t)
                              .to_event(3))

            # Insert into tensors
            # pred_latents_loc[:, i, :, :, :] = z_pred_loc
            # pred_latents_scale[:, i, :, :, :] = z_pred_scale
            pred_observations_loc[:, i, :, :, :] = emission_loc_t
            pred_observations_scale[:, i, :, :, :] = emission_scale_t

            z_prev = z_t

        if summed:
            observations_loss_loc = torch.sum(
                (pred_observations_loc - ground_truth[:, input_seq_len - pred_length:, :, :, :]) ** 2).detach().cpu().numpy()
            observations_loss_scale = torch.sum((pred_observations_scale - torch.ones(
                ground_truth[:, input_seq_len - pred_length:, :, :, :].shape).to(pred_batch.device)) ** 2).detach().cpu().numpy()
        else:
            observations_loss_loc = (
                (pred_observations_loc - ground_truth[:, input_seq_len - pred_length:, :, :, :]) ** 2).detach().cpu().numpy()
            observations_loss_scale = ((pred_observations_scale - torch.ones(
                ground_truth[:, input_seq_len - pred_length:, :, :, :].shape).to(pred_batch.device)) ** 2).detach().cpu().numpy()

    return pred_observations_loc, pred_observations_scale, observations_loss_loc, observations_loss_scale


if __name__ == '__main__':
    # Test
    input_channels = 1
    z_channels = 1
    emission_channels = [32, 16]
    transition_channels = 32
    encoder_channels = [16, 32]
    rnn_input_dim = 32
    rnn_channels = 32
    kernel_size = 3
    pred_length = 0
    input_length = 30
    width = 100
    height = 100

    input_tensor = torch.zeros(
        16, input_length, input_channels, width, height).cuda()
    input_tensor_mask = torch.ones(
        16, input_length, input_channels, width, height).cuda()
    input_tensor_reversed = reverse_sequences(input_tensor).cuda()

    pred_tensor = input_tensor[:, :25, :, :, :]
    pred_tensor_mask = input_tensor_mask[:, :25, :, :, :]
    pred_tensor_reversed = reverse_sequences(pred_tensor).cuda()

    ground_truth = input_tensor[:, 25:, :, :, :]

    pred_input_dim = 5

    dmm = DMM(input_channels=input_channels, z_channels=z_channels, emission_channels=emission_channels,
              transition_channels=transition_channels, encoder_channels=encoder_channels, rnn_input_dim=rnn_input_dim, rnn_channels=rnn_channels, kernel_size=kernel_size, height=height, width=width, pred_input_dim=5, num_layers=2, rnn_dropout_rate=0.0,
              num_iafs=0, iaf_dim=50, use_cuda=True)

    learning_rate = 0.01
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
    for i in range(100):
        loss = svi.step(input_tensor, input_tensor_reversed, input_tensor_mask)
        val_nll = svi.evaluate_loss(
            input_tensor, input_tensor_reversed, input_tensor_mask)
        print(val_nll)
        _, _, loss_loc, loss_scale = do_prediction(
            dmm, pred_tensor, pred_tensor_reversed, pred_tensor_mask, 5, ground_truth)
        print(loss_loc, loss_scale)
