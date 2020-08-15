from enum import Enum
import numpy as np
import scipy.stats as scistats
import matplotlib.pyplot as plt
from tqdm import tqdm


class Noise(Enum):
    NORMAL = 1
    CAUCHY = 2


def generate(
    n_steps=101,
    plate_width=10,
    plate_height=10,
    dx=0.1,
    dy=0.1,
    thermal_diffusivity=4,
    temperature_cool=0,
    temperature_hot=700,
    ring_params=(2, 5, 5),
    trans_stoch=None,
    trans_pdf=Noise.NORMAL,
    emis_stoch=None,
    emis_pdf=Noise.NORMAL
):
    """
    Generate diffusion heatmap with specified initialization parameters.


    Args:
        n_steps (int): Total timesteps.
        plate_width (int): Plate width.
        plate_height (int): Plate height.
        dx (float): x difference.
        dy (float): y difference.
        thermal_diffusivity (float): Thermal diffusivity constant.
        temperature_cool (float): Base temperature.
        temperature_hot (float): Hot temperature.
        ring_params (float, float, float): Positional and size parameters of heat ring.
            r (float): Radius of the ring.
            cx (float): x position of center of ring.
            cy (float): y position of center of ring.
        trans_stoch (loc (float), scale (float)): Add gaussian noise on ODE transition function.
        trans_pdf (Noise): Type of transition noise pdf
        emis_stoch (loc (float), scale (float)): Add gaussian noise on emission function.
        emis_pdf (Noise): Type of emission noise pdf

    Return:
        latents (np.array): Generated latent data.
        obs (np.array): Generated observation data.
        dt (float): Time difference between propagation.
    """
    # Plate width and height in pixel space
    nx, ny = int(plate_width / dx), int(plate_height / dy)
    nvar = nx * ny

    squared_dx = dx * dx
    squared_dy = dy * dy
    dt = squared_dx * squared_dy / \
        (2 * thermal_diffusivity * (squared_dx + squared_dy))  # Maximum time difference

    u_0 = temperature_cool * np.ones((nx, ny))
    x_0 = u_0.copy()

    # Initial conditions of ring
    # Inner radius r, width dr centered at (cx, cy) (mm)
    u_0 = add_ring(u_0, dx, dy, temperature=temperature_hot,
                   ring_params=ring_params)

    if emis_stoch is not None:
        x_0 = u_0 + generate_noise(emis_stoch, nx, ny)

    # Solve equation with time-difference
    latents = [u_0]
    obs = [x_0]
    for i in tqdm(range(n_steps)):
        u_0 = do_timestep(u_0, dt, squared_dx,
                          squared_dy, thermal_diffusivity)

        if trans_stoch is not None:
            u_0 = u_0 + generate_noise(trans_stoch, nx, ny, type=trans_pdf)

        if emis_stoch is not None:
            x_0 = u_0 + generate_noise(emis_stoch, nx, ny, type=emis_pdf)

        # Clip value below 0
        u_0 = np.clip(u_0, temperature_cool, temperature_hot + 300)
        x_0 = np.clip(x_0, temperature_cool, temperature_hot + 300)

        latents.append(u_0)
        obs.append(x_0)

    data = np.array(latents)
    obs = np.array(obs)

    return data, obs, dt


def add_ring(heatmap, dx, dy, ring_params, temperature=700):
    """
    Add ring of heat inside a specified heatmap.


    Args:
        heatmap (np.array): Heatmap that encompasses the plate.
        dx (int): x difference.
        dy (int): y difference.
        ring_params (float, float, float): Positional and size parameter of heat ring.
            r (float): Radius of the ring.
            cx (float): x position of center of ring.
            cy (float): y position of center of ring.
        temperature (float): Ring temperature.

    Return:
        heatmap (np.array): Plate with added ring of heat.
    """
    r, cx, cy = ring_params
    squared_r = r ** 2
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            squared_p = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if squared_p < squared_r:
                heatmap[i, j] = temperature

    heatmap = zero_border(heatmap)

    return heatmap


def do_timestep(u_prev, dt, squared_dx, squared_dy, thermal_diffusivity):
    """
    Propagate equation with forward-difference method in time and central-difference method in space.


    Args:
        u_prev (np.array): Previous heatmap state.

    Return:
        u (np.array): Propagated heatmap state.
    """
    nx = u_prev.shape[0]
    ny = u_prev.shape[1]
    u = np.empty((nx, ny))
    u[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + thermal_diffusivity * dt * (
        (
            u_prev[2:, 1:-1] - 2 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1]
        ) / squared_dx +
        (
            u_prev[1:-1, 2:] - 2 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2]
        ) / squared_dy
    )

    return u


def generate_noise(params, nx, ny, type=Noise.NORMAL, is_zero_border=True):
    """
    Generate noise according to specified pdf.


    Args:
        params (float, float): Mean and variance.
        nx (int): Width of the plate.
        ny (int): Height of the plate.
        type (Noise): Type of pdf of the noise
        zero_border (bool): Zeroes all the border pixels

    Return:
        noise (np.array): Generated noise.
    """
    nvar = nx * ny
    if type == Noise.NORMAL:
        noise = np.random.normal(
            params[0], params[1], nvar).reshape(nx, ny)
    elif type == Noise.CAUCHY:
        noise = scistats.cauchy.rvs(params[0], params[1], nvar).reshape(nx, ny)

    if is_zero_border:
        noise = zero_border(noise)

    return noise


def zero_border(plate):
    """
    Set the border of the plate into zero


    Args:
        plate (np.array): Plate to be zero-bordered

    Return:
        plate (np.array): Zero-bordered plate
    """
    ny, nx = plate.shape

    plate[0, :] = np.zeros(ny)
    plate[-1, :] = np.zeros(ny)
    plate[:, 0] = np.zeros(nx)
    plate[:, -1] = np.zeros(nx)

    return plate


def main():
    latent, obs = generate()


if __name__ == '__main__':
    main()
