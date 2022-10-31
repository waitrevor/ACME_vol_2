# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Name> Trevor
<Class> Section 2
<Date> 10/27/22
"""
from collections import deque
import networkx as nx
from matplotlib import pyplot as plt

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        #Adds node n into the set if not already present
        if n not in set(self.d.keys()):
            self.d[n] = set()

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        #Adds u and v in the graph if they are not already present
        self.add_node(u)
        self.add_node(v)
        self.d[u].add(v)
        self.d[v].add(u)



    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        self.d.pop(n)

        for i in self.d.keys():
            try:
                self.d[i].remove(n)
            except KeyError:
                pass


    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        self.d[u].remove(v)
        self.d[v].remove(u)


    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        V = []
        Q = deque()
        M = set()
        Q.append(source)
        M.add(source)

        while len(Q) != 0:
            node = Q.popleft()
            V.append(node)
            for i in self.d[node]:
                if i not in M:
                    M.add(i)
                    Q.append(i)
            
        return V

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        Q = deque()
        M = set()
        Q.append((source, list(source)))
        M.add(source)

        while len(Q) != 0:
            node, path = Q.popleft()
            if node is target:
                return path
            for i in self.d[node]:
                if i not in M:
                    M.add(i)
                    Q.append(i, path + list(i))
            


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.title = set()
        self.actor = set()
        self.graph = nx.Graph()

        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines:
                line = line.strip().split('/')
                self.graph.add_nodes_from(line)
                edge = []

                for name in line[1:]:
                    edge.append((line[0],name))
                    self.actor.add(name)

                self.graph.add_edges_from(edge)
                self.title.add(line[0])

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        return nx.shortest_path(self.graph, source, target), nx.shortest_path_length(self.graph, source, target) // 2

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        D = nx.shortest_path_length(self.graph, target)
        only_actors = {key: D[key] for key in self.actor}
        L = [item // 2 for item in only_actors.values()]
        plt.hist(L, bins=[i-0.5 for i in range(8)], log=True)
        plt.title(f"Kevin Bacon number of {target}")
        plt.ylabel("nubmer of actors")
        plt.xlabel("Kevin Bacon Number")
        plt.show()


