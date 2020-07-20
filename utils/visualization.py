import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from mpl_toolkits import mplot3d


def visualize(data, min_val, max_val, title, cbar_title, n_row=3, n_col=3, is_3d=False):
    """
    Visualize data in multigrid.


    Args:
        data (np.array): Data.
        min_val (float): Minimum value.
        max_val (float): Maximum value.
        title (str): Title of the visualization.
        n_row (int): Number of rows.
        n_col (int): Number of columns.
        is_3d (bool): Make 3D plot if true.
    """
    n_vis_image = n_row * n_col
    data_length, x_len, y_len = data.shape
    vis_indices = np.arange(1, data_length, int(data_length / n_vis_image))
    vis_data = data[vis_indices]

    x = np.outer(np.linspace(0, 10, x_len), np.ones(y_len))
    y = x.copy().T

    if not is_3d:
        fig, axs = plt.subplots(n_row, n_col)
        fig.suptitle(title)
        cmap = 'hot'

        images = []
        for i in range(n_row):
            for j in range(n_col):
                images.append(axs[i, j].imshow(
                    vis_data[i * n_col + j], cmap=cmap, vmin=min_val, vmax=max_val))
                axs[i, j].label_outer()
                axs[i, j].set_axis_off()

        cbar = fig.colorbar(images[0], ax=axs, orientation='horizontal')
        cbar.set_label(cbar_title)
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(x, y, data[0], cmap='hot', edgecolor='none')
        ax.set_title(title)

    plt.show()


def animate(data, min_val, max_val, dt, save_path=None, is_3d=False):
    """
    Animate data.


    Args:
        data (np.array): Data.
        min_val (float): Minimum value.
        max_val (float): Maximum value.
        dt (float): Time difference between frames.
        save_path (Path): Animation save path.
        is_3d (bool): Make 3D animation if true.
    """
    fig = plt.figure()
    cmap = 'hot'

    data_length, x_len, y_len = data.shape
    x = np.outer(np.linspace(0, 10, x_len), np.ones(y_len))
    y = x.copy().T

    if not is_3d:
        images = []
        for i in range(len(data)):
            image = plt.imshow(data[i], cmap=cmap, vmin=min_val,
                               vmax=max_val, animated=True)
            images.append([image])

        anim = animation.ArtistAnimation(fig, images, interval=50, blit=True,
                                         repeat_delay=1000)
    else:
        frn = len(data)
        fps = 30

        ax = fig.add_subplot(111, projection='3d')
        plot = [ax.plot_surface(x, y, data[0], cmap='hot')]
        ax.set_zlim(0, max_val)
        title = ax.text(1, 1, 0, s="",
                        transform=ax.transAxes, ha="center")

        def update_plot(frame_number, data, plot):
            title.set_text(u"Time = {:.2f} ms".format(dt * frame_number * 100))
            plot[0].remove()
            plot[0] = ax.plot_surface(
                x, y, data[frame_number], cmap=cmap)

            return plot[0], title

        anim = animation.FuncAnimation(
            fig, update_plot, frn, fargs=(data, plot), interval=1000/fps, blit=True)

    if save_path is not None:
        anim.save(save_path)
