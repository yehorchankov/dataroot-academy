import numpy as np
import matplotlib.pyplot as plt

def cluster_plot(X_dat, y_dat, x_coord=0, y_coord=1, 
                 labels=[], cen_x=[], cen_y=[], leg=True, 
                 noise=False, noise_i=-1, n_clust=0, seed=0):
    # Validate input data
    assert n_clust >= 0
    # Seed for data replication
    np.random.seed(seed)
    # Find number of clusters if not specified
    if n_clust == 0:
        denoise_lst = list(filter(lambda y: y >= 0, y_dat))
        n_clust = len(np.unique(denoise_lst))
    # normalize color, so each one represents from 30 to 230 RGB intensity
    # and its values lie between 30/255 to 230/255
    # this way the colors would not look very dark or light
    color = np.random.rand(n_clust,3)
    color = 30/255 + color*200/255
    for i in range(n_clust):
        # Name clusters if their labels specified
        if any(labels):
            label = labels[i]
        else:
            label = f'Cluster {i+1}'
        plt.scatter(X_dat[y_dat==i,x_coord],
                       X_dat[y_dat==i,y_coord],
                       s=50,
                       c=color[i],
                       marker='o',
                       label=label)
    # Plot centroids
    if any(cen_x):
        assert len(cen_x) == len(cen_y)
        plt.scatter(cen_x,
                   cen_y,
                   s=50,
                   marker='*',
                   c='orangered',
                   label='Centroid')
    # Plot noise
    if noise:
        plt.scatter(X_dat[y_dat == noise_i, x_coord], 
                    X_dat[y_dat == noise_i, y_coord],
                    c='lightgray', marker='o', s=10,
                    label='Noise')
    if leg:
        plt.legend()
    plt.tight_layout()
    plt.show()