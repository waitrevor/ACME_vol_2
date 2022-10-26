# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Name> Trevor Wai
<Class> Section 2
<Date> 10/20/22
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from statistics import mode


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    #Finds the norms of the rows
    norm_rows = la.norm(X - z, axis=1)
    #Find the position of the smallest norm
    paws = np.argmin(norm_rows)
    val = min(norm_rows)

    return X[paws], val


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        """Raises an error if x is not an np.ndarray and Initialize the left, right, pivot, and value"""
        if type(x) != np.ndarray:
            raise TypeError("x is not type np.ndarray")
        self.left = None
        self.right = None
        self.pivot = None
        self.value = x
        

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        #Checks to see if the data
        x = None
        try:
            x = self.find(data)
        except ValueError:
            pass

        if type(x) is KDTNode:
            raise ValueError("Data is already in tree")

        #Initializes the data as the root if the tree is empty
        if self.root == None:
            self.root = KDTNode(data)
            self.root.pivot = 0
            self.k = len(data)
            return

        if len(data) != self.k:
            raise ValueError("Data is not the right dimension")

        
        #Traverse the tree to see where to insert the data
        def _traverse(current):
            """Traverses through the tree to find where to insert the data into the tree"""
            piv = current.pivot

            #If the data is less then the current value go left
            if data[piv] < current.value[piv]:
                if current.left is None:
                    current.left = KDTNode(data)
                    current.left.pivot = (piv + 1) % self.k
                    return
                return _traverse(current.left)
            #If the data is greater than the current go right
            else:
                if current.right is None:
                    current.right = KDTNode(data)
                    current.right.pivot = (piv + 1) % self.k
                    return
                return _traverse(current.right)
            
            
        return _traverse(self.root)
            
        
            

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def _search(current, nearest, d):
            if current is None:
                return nearest, d
            x = current.value
            i = current.pivot
            if la.norm(x - z) < d:
                nearest = current
                d = la.norm(x - z)
            if z[i] < x[i]:
                nearest, d = _search(current.left, nearest, d)
                if z[i] + d >= x[i]:
                    nearest, d = _search(current.right, nearest, d)
            else:
                nearest, d = _search(current.right, nearest, d)
                if z[i] - d <= x[i]:
                    nearest, d = _search(current.left, nearest, d)
            return nearest, d
        node, d = _search(self.root, self.root, la.norm(self.root.value - z))
        return node.value, d


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        """Constructor that accepts n_neighbors and saves the value as an attribute"""
        self.num = n_neighbors

    def fit(self, X, y):
        """Uses SciPy KDTree with the data X and saves the tree and lables"""
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        """Finds the nearest neighbor to z and returns the most common label"""
        min_distance, index = self.tree.query(z, self.num)
        return mode([self.labels[i] for i in index])

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(float)
    y_test = data["y_test"]

    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train, y_train)
    correct = 0
    for i in range(len(X_test)):
        result = classifier.predict(X_test[i])
        if result == y_test[i]:
            correct += 1
    
    return correct / len(y_test)
