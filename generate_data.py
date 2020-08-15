from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import random
import torch
import hydra
import logging
from omegaconf import DictConfig

import data_generation.diffusion as diffusion


def get_pdf_type(pdf_type):
    """
    Get specific pdf type enumeration.


    Args:
        pdf_type (str): Type of the pdf in string.

    Return:
        pdf_type (diffusion.Noise): Type of the pdf in enumerated type.
    """
    if pdf_type == 'normal':
        return diffusion.Noise.NORMAL
    elif pdf_type == 'cauchy':
        return diffusion.Noise.CAUCHY

    raise Exception(
        'PDF type name error. Please specify a valid PDF type name.')


def exist_param(param, generated_params, tol=1e-4):
    """
    Check existence of parameters.


    Args:
        param ((float, float, float, float)): Generated parameters.
        generated_params ((float, float, float, float)[]): List of generated params
        tol (float): Maximum tolerated difference.

    Return:
        existence (bool): True if exist. False if otherwise.
    """
    for existing_params in generated_params:
        is_close = np.all(
            np.isclose(param, existing_params, rtol=tol)
        )

        if is_close:
            return True

    return False


def generate_param(radius_range, cx_range, cy_range, temp_range, generated_params):
    """
    Generate new ring heat parameters.


    Args:
        generated_params ((float, float, float, float)[]): List of generated params.

    Return:
        param ((float, float, float, float)): Generated param.
        generated_params ((float, float, float, float)[]): Updated list of generated params.
    """
    radius = np.random.uniform(radius_range[0], radius_range[1])
    cx = np.random.uniform(cx_range[0], cx_range[1])
    cy = np.random.uniform(cy_range[0], cy_range[1])
    temperature = np.random.uniform(temp_range[0], temp_range[1])
    param = (radius, cx, cy, temperature)
    while exist_param(param, generated_params):
        radius = np.random.uniform(radius_range[0], radius_range[1])
        cx = np.random.uniform(cx_range[0], cx_range[1])
        cy = np.random.uniform(cy_range[0], cy_range[1])
        temperature = np.random.uniform(temp_range[0], temp_range[1])
        param = (radius, cx, cy, temperature)

    generated_params.append(param)

    return param, generated_params


@hydra.main(config_path="./configs/data_generation/diffusion.yaml")
def main(cfg):
    # Assign parameters
    logging.info("Configurations")
    logging.info(cfg.pretty())
    params = cfg.parameters
    data_path = Path(params.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    n_simulations = params.n_simulations
    simulation_length = params.simulation_length
    n_samples = params.n_samples
    sample_length = params.sample_length
    n_samples = params.n_samples
    sample_time_difference = params.sample_time_difference

    temp_base = params.temp_base
    temp_low, temp_high = params.temp_low, params.temp_high
    temp_range = (temp_low, temp_high)
    radius_range = (params.radius_min, params.radius_max)
    cx_range = (params.center_x_min, params.center_x_max)
    cy_range = (params.center_y_min, params.center_y_max)

    trans_stoch = (params.transition_loc, params.transition_scale)
    trans_pdf = get_pdf_type(params.transition_noise_type)
    emis_stoch = (params.emission_loc, params.emission_scale)
    emis_pdf = get_pdf_type(params.emission_noise_type)

    # For parameter existence checking to avoid duplicates.
    generated_params = []
    generated_params_path = data_path / '../../generated_params.pkl'
    if generated_params_path.is_file():
        with open(generated_params_path, 'rb') as fin:
            generated_params = pickle.load(fin)

    # Generate data
    cur_index = 0
    for i in tqdm(range(n_simulations)):
        # Generate simulation
        param, generated_params = generate_param(
            radius_range, cx_range, cy_range, temp_range, generated_params)

        latent, obs, dt = diffusion.generate(
            n_steps=simulation_length,
            temperature_cool=temp_base,
            temperature_hot=param[3],
            ring_params=(param[0], param[1], param[2]),
            trans_stoch=trans_stoch,
            trans_pdf=trans_pdf,
            emis_stoch=emis_stoch,
            emis_pdf=emis_pdf
        )

        # Sample data from generated simulation
        start_points = random.sample(
            range(len(latent) - sample_length * sample_time_difference), n_samples)
        logging.info(
            'Start sample points of simulation {}: {}'.format(i, start_points))

        for start in start_points:
            indices = np.arange(
                start, start + (sample_length * sample_time_difference), sample_time_difference)
            temp_data = {
                'latent': torch.Tensor(latent[indices, :, :]),
                'observation': torch.Tensor(obs[indices, :, :])
            }

            with open(data_path / '{}.pkl'.format(cur_index), 'wb') as fout:
                pickle.dump(temp_data, fout)

            cur_index += 1

    logging.info("Generated {} sequences.".format(cur_index))

    with open(generated_params_path, 'wb') as fout:
        pickle.dump(generated_params, fout)


if __name__ == '__main__':
    main()
