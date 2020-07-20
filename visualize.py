from utils import visualization
import pickle
import argparse
from pathlib import Path
import torch
import numpy as np


def main(args):
    loadpath = Path(args.loadpath)
    savepath = Path(args.savepath)
    with open(loadpath, 'rb') as fin:
        data = pickle.load(fin)

    dt = 0.000625 * 3
    temp_low = 0
    temp_high = 700

    try:
        data = data.numpy()
    except:
        data = data['observation'].numpy()
    print(data.shape)

    if not args.anim:
        visualization.visualize(
            data=data,
            title='Diffusion',
            cbar_title='$T$ / K',
            min_val=temp_low,
            max_val=700,
            n_row=3,
            n_col=10,
            is_3d=True
        )

    if args.anim:
        visualization.animate(data, min_val=temp_low,
                              max_val=700, dt=dt, save_path=savepath / 'animation_{}.mp4'.format(loadpath.name.split('.')[0]), is_3d=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated data")
    parser.add_argument('--loadpath', '-lp', type=str, help="Data load path")
    parser.add_argument('--savepath', '-sp', type=str, help="Data save path")
    parser.add_argument('--anim', '-a', action='store_true',
                        help='create animation')
    args = parser.parse_args()
    main(args)
