# binary_trees.py
"""Volume 2: Binary Trees.
<Name> Trevor Wai
<Class> Section 2
<Date> 10/18/22
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
from time import perf_counter
from random import sample
import numpy as np



class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        #Function to traverse the tree
        def _traverse(current):
            #If the list is empty or if no such node exists
            if current is None:
                raise ValueError("No such node in the list")
            #Returns the node with same value as the given data
            if current.value == data:
                return current
            #Recursively searches the tree if the node is not yet found
            else:
                return _traverse(current.next)
        
        return _traverse(self.head)


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        #If the list is empty
        if self.root == None:
                self.root = BSTNode(data)
                return
        
        #Function to traverse the tree
        def _traverse(current):

            #If the data already is in the tree
            if data == current.value:
                raise ValueError("No duplicate data allowed")

            #If the data is less than the current go left
            elif data < current.value:
                #If the left doesn't have any more nodes
                if current.left is None:
                    current.left = BSTNode(data)
                    current.left.prev = current
                else:
                    return _traverse(current.left)
            #If the data is greater than the current go right
            elif data > current.value:
                #If the right doesn't have any more nodes
                if current.right is None:
                    current.right = BSTNode(data)
                    current.right.prev = current
                else:
                    return _traverse(current.right)

        return _traverse(self.root)

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        target = self.find(data)

        #Target is a leaf
        if target.left is None and target.right is None:

            #The target is the root with no children
            if target is self.root:
                self.root = None
                return

            #Target is left child
            elif target.prev.left is target:
                target.prev.left = None

            #Target is right child
            else:
                target.prev.right = None

        #Target has two children
        elif target.left is not None and target.right is not None:
            pred = target.left

            #Finds the right most left child
            while pred.right is not None:
                pred = pred.right
            pred_val = pred.value

            #Removes the inorder predecessor and replaces the target
            self.remove(pred_val)
            target.value = pred_val
        
        #Target has one child
        elif bool(target.left is None) != bool(target.right is None):

            #Target has a left child
            if target.left is not None:
                child = target.left

                #If the target is the root
                if target is self.root:
                    self.root = child
                    child.prev = None

                #If the target is the left child
                elif target is target.prev.left:
                    parent = target.prev
                    parent.left = child
                    child.prev = parent

                #If the target is the right child
                elif target is target.prev.right:
                    parent = target.prev
                    parent.right = child
                    child.prev = parent

            #Target has a right child
            elif target.right is not None:
                child = target.right

                #If the target is the root
                if target is self.root:
                    self.root = child
                    child.prev = None

                #If the target is the left child
                elif target is target.prev.left:
                    parent = target.prev
                    parent.left = child
                    child.prev = parent

                #If the target is the right child
                elif target is target.prev.right:
                    parent = target.prev
                    parent.right = child
                    child.prev = parent

                




    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    with open("english.txt") as inFile:
        lines = inFile.readlines()

    #Initializes variables
    domain = 2**np.arange(3,11)

    makeList = []
    makeBST = []
    makeAVL = []

    readList = []
    readBST = []
    readAVL = []

    #Loops through the domain
    for n in domain:

        #Initializes variables
        theList = SinglyLinkedList()
        theBST = BST()
        theAVL = AVL()
        theWords = sample(lines, 5)

        #Records how long it takes to make the Linked List
        start = perf_counter()
        for word in theWords:
            theList.append(word)
        end = perf_counter() - start
        makeList.append(end)

        #Records how long it takes to make the BST
        start = perf_counter()
        for word in theWords:
            theBST.insert(word)
        end = perf_counter() - start
        makeBST.append(end)

        #Records how long it takes to make the AVL
        start = perf_counter()
        for word in theWords:
            theAVL.insert(word)
        end = perf_counter() - start
        makeAVL.append(end)

        #Pick 5 words
        subset = []
        for k in range(5):
            subset.append(np.random.choice(list(theWords)))

        #Records how long it takes to search the Linked List
        start = perf_counter()
        for word in subset:
            theList.iterative_find(word)
        end = perf_counter() - start
        readList.append(end)

        #Records how long it takes to search the BST
        start = perf_counter()
        for word in subset:
            theBST.find(word)
        end = perf_counter() - start
        readBST.append(end)

        #Records how long it takes to search the AVL
        start = perf_counter()
        for word in subset:
            theAVL.find(word)
        end = perf_counter() - start
        readAVL.append(end)

    #Plots how long it takes to make the linked list, bst and avl
    plt.subplot(121)
    plt.plot(domain, makeList)
    plt.plot(domain, makeBST)
    plt.plot(domain, makeAVL)
    plt.title("Making the List, BST, AVL")
    plt.xlabel("Number of elements")
    plt.ylabel("Time")
    plt.legend(["LL", "BST", "AVL"])

    #Plots how long it takes to seach the linked list, bst, avl
    plt.subplot(122)
    plt.loglog(domain, readList)
    plt.loglog(domain, readBST)
    plt.loglog(domain, readAVL)
    plt.title("Read the List, BST, AVL")
    plt.xlabel("Number of elements")
    plt.ylabel("Time")
    plt.legend(["LL", "BST", "AVL"])

    plt.tight_layout()
    plt.show()
