from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import random
import torch

import data_generation.diffusion as diffusion


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


def main(args):
    # Generate data
    data_path = Path(args.savepath)
    data_path.mkdir(parents=True, exist_ok=True)

    N = 25
    temp_base = 0
    temp_low, temp_high = 500, 700
    temp_range = (temp_low, temp_high)
    radius_range = (0.5, 5)
    cx_range = (0, 10)
    cy_range = (0, 10)
    trans_stoch = (0, 3)
    emis_stoch = (0, 10)
    emis_pdf = diffusion.Noise.CAUCHY
    n_steps = 2001
    sample_length = 50

    cur_index = 0

    # For parameter existence checking to avoid duplicates.
    generated_params = []
    generated_params_path = Path('./generated_params.pkl')
    if generated_params_path.is_file():
        with open(generated_params_path, 'rb') as fin:
            generated_params = pickle.load(fin)

    # Generate data
    for i in tqdm(range(N)):
        param, generated_params = generate_param(
            radius_range, cx_range, cy_range, temp_range, generated_params)

        latent, obs, dt = diffusion.generate(
            n_steps=n_steps,
            temperature_cool=temp_base,
            temperature_hot=param[3],
            ring_params=(param[0], param[1], param[2]),
            trans_stoch=trans_stoch,
            emis_stoch=emis_stoch
        )

        # # Sample time-series from generated steps
        n_sample = 30
        diff_index = 3
        start_points = random.sample(
            range(len(latent) - sample_length * diff_index), n_sample)

        print(start_points)

        for start in start_points:
            indices = np.arange(
                start, start + (sample_length * diff_index), diff_index)
            temp_data = {
                'latent': torch.Tensor(latent[indices, :, :]),
                'observation': torch.Tensor(obs[indices, :, :])
            }

            with open(data_path / '{}.pkl'.format(cur_index), 'wb') as fout:
                pickle.dump(temp_data, fout)

            cur_index += 1

    print("Generated {} sequences.".format(cur_index))

    with open(generated_params_path, 'wb') as fout:
        pickle.dump(generated_params, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate data for training")
    parser.add_argument('--savepath', '-sp',
                        default='./data/diffusion/cauchy_10', type=str)
    args = parser.parse_args()
    main(args)
