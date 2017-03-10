# %load "../models/Utilities.py"
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.cm import jet
from IPython.display import Image
import pydotplus
import itertools


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


def plot_decision_tree(X, y, dtree, path="tree.dot"):
    """ Visualize Decision Tree structures
    
    INPUT:
    ------
    X:      Dataframe containing the attributes the decision tree used
    y:      Dataframe containing the class labels the decision tree used
    dtree:  The trained decision tree
    path:   Path to store the .dot tree file, default='tree.dot'
    
    
    OUTPUT:
    -------
    image:  Image of the visualized decision tree
    
    
    DESCRIPTION:
    ------------
    The visualization of the decision tree is based on the following steps:
    - export the tree structure as a .dot
    - create a png based on the .dot
    - return an image based on the png
    
    """
    export_graphviz(dtree,
                    out_file=path,
                    feature_names=X.columns,
                    class_names=np.unique(dfp.inverse_transform_label(y)),
                    filled = True,
                    rounded = True)
    graph = pydotplus.graph_from_dot_file(path)
    graph.write_png(path)
    return Image(graph.create_png(), width=500)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Method taken from:
    http://scikit-learn.org/stable/auto_examples/
    model_selection/plot_confusion_matrix.html#
    sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return(plt)
