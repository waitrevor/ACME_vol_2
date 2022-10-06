# linked_lists.py
"""Volume 2: Linked Lists.
<Name> Trevor Wai
<Class> Section 2
<Date> 10/6/22
"""


# Problem 1
from tkinter import N
from unittest.mock import NonCallableMagicMock


class Node:
    """A basic node class for storing data of type int, float, or str."""
    def __init__(self, data):
        """Store the data in the value attribute of type int, float or string.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        #Stores data if it is an int float or string otherwise raises an error
        if type(data) is int:
            self.value = data
        elif type(data) is float:
            self.value = data
        elif type(data) is str:
            self.value = data
        else:
            raise TypeError('Data is not of type int, float, or str.')



class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        #Assings the current node as the head
        node = self.head

        #Loops through the linked list to find the node
        while (node is not self.tail):
            #Tests to see if the value of the node is equal to the data
            if node.value == data:
                return node
            else:
                node = node.next
        
        #Catchs if the tail is the node else raises errors
        if node.value == data:
            return node
        elif self.head == None:
            raise ValueError('List is empty')
        elif node == self.tail:
            raise ValueError('Node not found')
        

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        #Assings node to be the head
        node = self.head
        count = 0

        #Loops through the linked list
        while(node is not self.tail):
            if count == i:
                return node
            #Updates the counter and the node
            count += 1
            node = node.next
        
        count +=1
        #Raises an error if the input is out of range
        if i < 0 or i >= count:
            raise IndexError('i is negative or greater than or equal to the current number of nodes')
        
        return node

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        node = self.head
        count = 0

        #Loops through the linked list
        while(node is not self.tail):
            #Updates the counter and the node
            count += 1
            node = node.next
        #Updates the counter to so it has the number of nodes
        count +=1

        return count

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        node = self.head
        l = []
        count = 0

        while(node is not self.tail):
            l.append(node.value)

            #Updates the counter and the node
            count += 1
            node = node.next
        #Updates the counter to so it has the number of nodes
        l.append(self.tail.value)
        return str(l)

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        #Raises an error if the list is empty
        if self.head == None:
            raise ValueError('List is empty')
        #Finds the target node
        target = self.find(data)
        #If the node is at the front then deletes the head
        if target is self.head:
            self.head = target.next
            self.head.prev = None
        #Ifthe node is at the tail then deletes the tail
        elif target is self.tail:
            self.tail = target.prev
            self.tail.next = None
        else:
            #Makes the node in front of target point to the node after target and vise versa
            target.prev.next = target.next
            target.next.prev = target.prev
        

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """

        #Checks to see if the index is at the end of the list
        if index == len(self):

            self.append(data)
            return
        #If the index isn't at the end then declares new nodes
        node = self.get(index)
        prev_node = node.prev
        new_node = LinkedListNode(data)
        
        #If the data is being added at the head
        if index == 0:
            #Adds the data to the head
            self.head = new_node
            new_node.next = node
            node.prev = new_node
            
        
        #Raises an error if the index is out of range
        elif index > len(self) or index < 0:
            raise IndexError('Index is out of range')
        #Inserts a node anywhere else in the linkedlist
        else:
            new_node.next = node
            node.prev = new_node
            prev_node.next = new_node
            new_node.prev = prev_node


            



# Problem 6: Deque class.
class Deque(LinkedList):
    """Class Deque that inherits form LinkedList but behaves like a deque"""

    def pop(self):
        """Pops the tail and returns value of the popped node"""
        #Raises an error if the list is empty
        if self.head == None:
            raise ValueError('List is empty')
        #Pops the tail
        else:
            node = self.tail
            LinkedList.remove(self, node)
            return node.value

    def popleft(self):
        """Pops the head and returns the value of the popped node"""
        #Raises an error if the list is empty
        if self.head == None:
            raise ValueError('List is empty')
        #Pops the head
        else:
            node = self.head
            LinkedList.remove(self, node)
        return node.value

    def appendleft(self, data):
        """inserts data at the head of the deque"""
        LinkedList.insert(self, 0, data)
    
    def remove(*args, **kwargs):
        """Raises an error if the remove function was attempted to be used"""
        raise NotImplementedError('Use pop() or popleft() for removal')
    
    def insert(*args, **kwargs):
        """Raises an error if the insert function was attempted to be used"""
        raise NotImplementedError('Use append() or appendleft() for insertion')

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    #Opens the file and reads the lines
    with open(infile, 'r') as file:
        lines = file.readlines()
    #Writes to an outfile that reverses the lines
    with open(outfile, 'w') as out:
        lines[-1] += '\n'
        new_list = lines[::-1]
        output = ''.join(new_list)
        out.write(output.strip())



#Testing
# my_list = LinkedList()
# for i in range(10):
#     my_list.append(i)

# print(my_list)
# my_list.insert(0,9)
# print(my_list)
prob7('english.txt', 'outfile.txt')