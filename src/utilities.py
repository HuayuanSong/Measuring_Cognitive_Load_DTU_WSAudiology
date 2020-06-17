# Fagprojekt
# Utility functions

# Load dependencies
from matplotlib.pyplot import cycler
import numpy as np
import matplotlib.cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Function to do color map for line plots
def get_cycle(cmap, N = None, use_index = "auto"):
    """
    Input:
        cmap = Color map using matplotlib's colormaps

    Output:
        Color cycler, look at example for usage information
    """

    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)

# Helper function, used to browse through the matlab ndarray structure
def lookAt(obj, extra = False):
    """
    Input:
        obj = Numpy ndarray object

    Output:
        Information about the array
    """

    shape = str(obj.shape)
    length = str(len(obj))
    type_ = str(type(obj))
    print("Shape: %s, Length: %s, Type: %s" %(shape, length, type_))

    if extra == True:
        first = str(obj[0])
        last = str(obj[-1])
        print("First element: %s, Last element: %s" %(first, last))