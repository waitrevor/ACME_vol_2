# shell2.py
"""Volume 3: Unix Shell 2.
<Name> Trevor Wai
<Class> Section 2
<Date> 11/16/22
"""

import os
from glob import glob
import subprocess
import numpy as np

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.

    Returns:
        matched_files (list): list of the filenames that matched the file
               pattern AND the target string.
    """
    #Initialize Variables use glob to find the files within the directory
    matched_files = []
    match_pattern = glob(f'**/{file_pattern}', recursive=True)

    #Loops through the files and looks for the target string in the files.
    for file in match_pattern:
        with open(file, 'r') as infile:
            if target_string in infile.read():
                matched_files.append(file)

    return matched_files


# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    #Searches and sorts the filenames
    infiles = glob('**/*.*', recursive=True)
    sizes = [os.path.getsize(file) for file in infiles]
    infiles = list(np.array(infiles)[np.argsort(sizes)][::-1][:n])

    #Counts the number of lines of the smallest file
    num_lines = subprocess.check_output(['wc', '-1', infiles[-1]]).decode()

    #Writes the number of lines to a file
    with open('smallest.txt', 'w') as outfile:
        outfile.write(num_lines[:num_lines.find(' ')])

    return infiles
    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter
