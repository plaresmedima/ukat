import os
import numpy as np
import matplotlib.pyplot as plt
import mdreg

from ukat.mapping.t1 import T1


def t1_maps(parameters=2, molli=False):

    # fetch data
    data = mdreg.fetch('MOLLI')
    array = data['array'][:, :, 0, :]
    TI = np.array(data['TI'])

    # Calculate uncorrected T1-map
    # Note: using absolute value here to ensure correct comparison
    map = T1(np.abs(array), TI, np.eye(4), parameters=parameters, molli=molli)
    file = 't1_uncorr_' + str(parameters) + ('_molli' if molli else '')
    np.save(os.path.join(os.getcwd(), file), map.t1_map)

    # Calculate corrected T1-map
    map = T1(array, TI, np.eye(4), parameters=parameters, molli=molli, mdr=True)
    file = 't1_corr_' + str(parameters) + ('_molli' if molli else '')
    np.save(os.path.join(os.getcwd(), file), map.t1_map)


def plot_t1_maps(parameters=2, molli=False):

    # Plot corrected and uncorrected side by side
    file = 't1_uncorr_' + str(parameters) + ('_molli' if molli else '')
    t1_uncorr = np.load(os.path.join(os.getcwd(), file+'.npy'))
    file = 't1_corr_' + str(parameters) + ('_molli' if molli else '')
    t1_corr = np.load(os.path.join(os.getcwd(), file)+'.npy')
    fig, ax = plt.subplots(figsize=(10, 6), ncols=2, nrows=1)
    for i in range(2):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
    ax[0].set_title('T1-map without MDR')
    ax[1].set_title('T1-map with MDR')
    im = ax[0].imshow(t1_uncorr.T, cmap='gray', vmin=0, vmax=2000)
    im = ax[1].imshow(t1_corr.T, cmap='gray', vmin=0, vmax=2000)
    fig.colorbar(im, ax=ax.ravel().tolist())
    file = 't1_mdr_'+ str(parameters) + ('_molli' if molli else '')
    plt.savefig(os.path.join(os.getcwd(), file))
    plt.show()


if __name__ == '__main__':

    t1_maps(parameters=2)
    plot_t1_maps(parameters=2)

    t1_maps(parameters=3)
    plot_t1_maps(parameters=3)

    t1_maps(molli=True)
    plot_t1_maps(molli=True)
