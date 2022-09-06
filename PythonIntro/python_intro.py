# python_intro.py
"""Python Essentials: Introduction to Python.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/1/22
"""


# Problem 2
from codecs import latin_1_decode


def sphere_volume(r):
    """ Return the volume of the sphere of radius 'r'.
    Use 3.14159 for pi in your computation.
    """
    #value of pi
    pi = 3.14159

    #formula for volume
    volume = 4 / 3 * pi * r

    return volume


# Problem 3
def isolate(a, b, c, d, e):
    """ Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    #prints a,b,c with 5 spaces of separation
    print(a, b, c, sep = '     ', end= ' ')

    #prints the rest of the parameters with only 1 space
    print(d, e, sep=' ')



# Problem 4
def first_half(my_string):
    """ Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    #Length of the string
    a = len(my_string)

    #Find half of the length
    b = a // 2

    #Find half the string
    half_string = my_string[:b]

    #returns half the string
    return half_string



def backward(my_string):
    """ Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """

    #Reverses the string
    backward_string = my_string[::-1]

    #returns the backwards string
    return backward_string


# Problem 5
def list_ops():
    """ Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    #Creates List
    my_list = ["bear", "ant", "cat", "dog"]

    #Appends the List
    my_list.append("eagle")

    #Changes index 2 to fox
    my_list[2] = "fox"

    #pops index 1
    my_list.pop(1)

    #puts the list into reverse aphabetical order
    my_list.sort(reverse=True)

    #Replaces eagle with hawk
    a = my_list.index("eagle")
    my_list[a] = "hawk"

    #adds hunter to the end of the last entry
    string = "hunter"
    b = len(my_list) - 1
    temp = my_list[b]
    my_list[b] = temp + string

    #returns the list
    return my_list
    


# Problem 6
def pig_latin(word):
    """ Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    #finds if the first letter is a vowel
    if (word[0] in "aeiou"):
        #if the first letter is a vowel returns word + hay
        return word + "hay"
    #if the word doesn't begin with a vowel
    else:
        #takes the first letter and adds ay to the end
        latin = word[0] + "ay"
        #take the rest of the word without the first letter
        pig_word = word[1:]
        #puts the two parts of the word together to translate the word into pig latin
        return pig_word + latin



# Problem 7
def palindrome():
    """ Find and retun the largest panindromic number made from the product
    of two 3-digit numbers.
    """
    #defines my panidromic variable
    n = 0

    #for loops that iterate from 999 down to be multiplied together
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):
            #x is the product of i and j
            x = i * j
            #check to make sure that we are not finding larger palindromes
            if x > n:
                #creates a string so we can test using our backwards function to see if the number is a palindrome
                number = str(x)
                if number == backward(number):
                    #sets n = to the palindrome
                    n = i * j

    #returns the largest palindrome
    return n

# Problem 8
def alt_harmonic(n):
    """ Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """

    L = [(-1) ** (i + 1) / (i) for i in range(1,n)]
    ans = sum(L)
    return ans
    

# Problem 1 (write code below)
if __name__ == "__main__":
    print("Hello, world!") 


#Testing goes here

print(alt_harmonic(500000))