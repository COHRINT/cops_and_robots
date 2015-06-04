from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_multisurface(X, Y, Z, ax, cmaps=None, min_alpha=0.6, **kwargs):
    num_surfs = Z.shape[2]

    if cmaps == None:
        cmaps = ['Greys', 'Reds', 'Purples', 'Oranges', 'Greens', 'Blues',
                   'RdPu']
        while num_surfs > len(cmaps):
                cmaps += cmaps

    #<>TODO: Include customizable c_max and c_min for color
    z_max = np.zeros(num_surfs)
    for i in range(num_surfs):
        z_max[i] = np.nanmax(Z[:,:,i])

    z_min = np.zeros(num_surfs)
    for i in range(num_surfs):
        z_min[i] = np.nanmin(Z[:,:,i])

    # Set color values
    C = np.zeros_like(Z, dtype=object)
    for z_i in range(num_surfs):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                cmap = cmaps[z_i]
                z_norm = (Z[i, j, z_i] - z_min[z_i]) / (z_max[z_i] - z_min[z_i])
                color = list(plt.get_cmap(cmap)(z_norm * 0.6 + 0.2) )
                color[3] = np.max([z_norm * 0.7 + 0.2, min_alpha])  # set alpha
                C[i, j, z_i] = color

    # Create a transparent bridge region
    X_bridge = np.vstack([X[-1,:], X[0,:]])
    Y_bridge = np.vstack([Y[-1,:], Y[0,:]])
    Z_bridge = np.zeros((X_bridge.shape[0], Y_bridge.shape[1], num_surfs - 1))
    C_bridge = np.empty_like(Z_bridge, dtype=object)
    for z_i in range(num_surfs - 1):
        Z_bridge[:, :, z_i] = np.vstack([Z[-1, :, z_i], Z[0, :, z_i + 1]])
    C_bridge.fill((1,1,1,0)) # RGBA colour, only the last component matters.

    # Join each two-pair of surfaces surfaces flipping one of them (using also the bridge)
    X_full = np.vstack([X, X_bridge, X])
    Y_full = np.vstack([Y, Y_bridge, Y])
    Z_full = np.vstack([Z[:, :, 0], Z_bridge[:, :, 0], Z[:, :, 1]])
    C_full = np.vstack([C[:, :, 0], C_bridge[:, :, 0], C[:, :, 1]])

    # Join any additional surfaces
    z_i = 1
    while z_i + 1 < num_surfs:
        X_full = np.vstack([X_full, X_bridge, X])
        Y_full = np.vstack([Y_full, Y_bridge, Y])    
        Z_full = np.vstack([Z_full, Z_bridge[:, :, z_i], Z[:, :, z_i + 1]])
        C_full = np.vstack([C_full, C_bridge[:, :, z_i], C[:, :, z_i + 1]])
        z_i += 1



    surf_full = ax.plot_surface(X_full, Y_full, Z_full, linewidth=0,
                                facecolors=C_full, antialiased=True, **kwargs)

    return surf_full

if __name__ == '__main__':
    from scipy.special import erf

    X = np.arange(-5, 5, 0.3)
    Y = np.arange(-5, 5, 0.3)
    X, Y = np.meshgrid(X, Y)

    Z1 = np.empty_like(X)
    Z2 = np.empty_like(X)
    Z3 = np.empty_like(X)
    Z4 = np.empty_like(X)

    for i in range(len(X)):
      for j in range(len(X[0])):
        z1 = 0.5*(erf((+X[i,j]+Y[i,j] - 2)*0.5) +1)
        z2 = 0.5*(erf((+X[i,j]-Y[i,j] - 2)*0.5) +1)
        z3 = 0.5*(erf((-X[i,j]+Y[i,j] - 2)*0.5) +1)
        z4 = 0.5*(erf((-X[i,j]-Y[i,j] - 2)*0.5) +1)
        Z1[i,j] = z1
        Z2[i,j] = z2
        Z3[i,j] = z3
        Z4[i,j] = z4

    Z = np.dstack((Z1, Z2, Z3, Z4))

    fig = plt.figure(figsize=(10,8))
    ax = fig.gca(projection='3d')


    surf = plot_multisurface(X, Y, Z, ax, cstride=1, rstride=1)
    plt.show()