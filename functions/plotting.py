import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable




def plot_colored_circles(name, colors, save_file, radius=0.4, spacing=1.0):
    """
    Plot N filled circles side-by-side, given a list of N colors. (Only N = 3 is used.)

    Parameters
    ----------
    colors : list of str
        List of color names or hex codes, one per circle.
    radius : float
        Radius of each circle.
    spacing : float
        Distance between centers of adjacent circles.
    """
    eps = 0.1
    
    N = len(colors)
    fig, ax = plt.subplots(figsize=(2,2))

    for i, color in enumerate(colors):
        circle = plt.Circle((i * spacing, 0), radius, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(- eps -radius, spacing * (N - 1) + radius + eps)
    ax.set_ylim(-radius - 0.1, radius + 0.1)
    ax.axis('off')  # Hide axes
    save_file(name)
    plt.show()
    
    
    
# Visualize 3D eigenvector as 3 colored circles
def plot_eigenvector(name, v, save_file):
    #cmap = plt.get_cmap('RdYlGn')   # unused colormaps
    #cmap = plt.get_cmap('Greys_r')
    cmap = plt.get_cmap('bwr_r')
    
    v_norm = (v + 1)/2; c = cmap(v_norm)

    plot_colored_circles(name, c, save_file)
    return


# Plot colorbar used for eigenvectors (this ONLY plots the colorbar)
def plot_colorbar(name, save_file):
        
    # Create the colormap and normalizer
    #cmap = plt.get_cmap('RdYlGn')    # unused colormaps
    #cmap = plt.get_cmap('Greys_r')
    cmap = plt.get_cmap('bwr_r')
    norm = Normalize(vmin=-1, vmax=1)


    # Create a color bar to show mapping
    fig, ax = plt.subplots(figsize=(3, 0.5))
    fig.subplots_adjust(bottom=0.5)
    
    # Use ScalarMappable to connect data values to colors
    cb = plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal', ticks=[-1, 0, 1]
    )
    cb.ax.tick_params(labelsize=25)
    save_file(name)
    plt.show()
    return 
