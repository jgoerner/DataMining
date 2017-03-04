# %load "../models/Utilities.py"
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.cm import jet

def plot_decision_regions(classifier, data, resolution=1000, legend=True, centroids=None):
    """Plotting decision regions given a classifier and data
    
    INPUT:
    ------
    classifier:  classifier, that inherits from the sklearn base packages 
    data:        data that should be visualized
    resolution:  resolution of the decision regions, default=1000
    legend:      bool if a legend should be added, default=True
    centroids:   plot cluster centroids (if applicable)
    
    OUTPUT:
    -------
    plot:        the plot objects, to be further annotated
    
    DESCRIPTION:
    ------------
    Based on a already learned classifier, this method plots:
    - the input + predicted label as a scatter plot
    - the decision regions per class as an area plot
    
    This method is derived from Raschka[2016], "Pyhton Machine Learning".
    All cudos to this outstanding machine learner!
    """
    
    # prepare the colormap
    colors = ('red', 'blue', 'green', 'black', 'purple', 'orange', 'yellow', 'pink', 'cyan', 'white')    
    if (len(np.unique(classifier.labels_)) > len(colors)):
        print("Not enough colors specified, please adjust colormaps")
        return
    else:
        n_cluster = len(np.unique(classifier.labels_))
        colmap = ListedColormap(colors[:n_cluster])
        bounds = np.arange(n_cluster + 1)-0.5
        norm = BoundaryNorm(bounds, n_cluster)

    # scatter plot per cluster
    for idx, label in enumerate(np.unique(classifier.labels_)):
        cluster = data.ix[classifier.labels_ == label]
        plt.scatter(cluster.iloc[:,0],
                    cluster.iloc[:,1],
                    c=colmap(idx),
                    label="c - {}".format(label))
    
    # decision regions
    x = np.linspace(data.iloc[:,0].min() - 0.25, data.iloc[:,0].max() + 0.25, resolution)
    y = np.linspace(data.iloc[:,1].min() - 0.25, data.iloc[:,1].max() + 0.25, resolution)
    xx, yy = np.meshgrid(x, y)
    z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    zz = z.reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=.15, cmap=colmap, norm=norm)
    
     # optional centroids
    if centroids:
        plt.scatter(classifier.cluster_centers_.T[0],
                    classifier.cluster_centers_.T[1],
                    marker="o",
                    facecolors='none',
                    edgecolors='k',
                    s=100,
                    label="centroid")
            
    # optional legend
    if legend:
        plt.legend(loc="best")
    
    # return plot object
    return(plt)