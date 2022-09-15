# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/15/22
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """ Create an (n x n) array of values randomly sampled from the standard
    normal distribution. Compute the mean of each row of the array. Return the
    variance of these means.

    Parameters:
        n (int): The number of rows and columns in the matrix.

    Returns:
        (float) The variance of the means of each row.
    """

    var = np.var(np.mean(np.random.normal(size=(n,n)), axis = 1))

    return var

def prob1():
    """ Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    n = 100
    array = []
    while n <= 1000:
        array.append(var_of_means(n))
        n += 100

    plt.plot(array)
    plt.show()


# Problem 2
def prob2():
    """ Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    sinRange = np.sin(x)
    cosRange = np.cos(x)
    arctanRange = np.arctan(x)
    plt.plot(x, sinRange)
    plt.plot(x, cosRange)
    plt.plot(x, arctanRange)
    plt.show()


# Problem 3
def prob3():
    """ Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x1 = np.linspace(-2, 1, 50, False)
    x2 = np.linspace(6, 1, 50, False)
    y1 = 1 / (x1 - 1)
    y2 = 1 / (x2 - 1)
    plt.plot(x1, y1, 'm--', linewidth = 4)
    plt.plot(x2, y2, 'm--', linewidth = 4)
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.show()



# Problem 4
def prob4():
    """ Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi], each in a separate subplot of a single figure.
        1. Arrange the plots in a 2 x 2 grid of subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, 2 * np.pi, 100)
    plt.subplot(224)
    #Top left
    ax1 = plt.subplot(221)
    ax1.plot(x, np.sin(x), 'g-')
    ax1.set_xlim([0, 2 * np.pi])
    ax1.set_ylim([-2, 2])
    #Top right
    ax2 = plt.subplot(222)
    ax2.plot(x, np.sin(2 * x), 'r--')
    ax2.set_xlim([0, 2 * np.pi])
    ax2.set_ylim([-2, 2])
    #Bottom left
    ax3 = plt.subplot(223)
    ax3.plot(x, 2 * np.sin(x), 'b--')
    ax3.set_xlim([0, 2 * np.pi])
    ax3.set_ylim([-2, 2])
    #Bottom Right
    ax4 = plt.subplot(224)
    ax4.plot(x, 2 * np.sin(2 * x), 'm:')
    ax4.set_xlim([0, 2 * np.pi])
    ax4.set_ylim([-2, 2])
    plt.tight_layout()
    plt.show()


# Problem 5
def prob5():
    """ Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    fars = np.load('FARS.npy')
    x = np.reshape(fars[:,1:2], (1,-1)).tolist()[0]
    y = np.reshape(fars[:,2:], (1,-1)).tolist()[0]

    plt.subplot(122)

    ax1 = plt.subplot(121)
    ax1.plot(x, y, 'k,')
    ax1.axis("equal")

    ax2 = plt.subplot(122)
    hours = np.reshape(fars[:,:1], (1,-1)).tolist()[0]
    
    ax2.hist(hours, bins=24)
    plt.tight_layout()
    plt.show()

# Problem 6
def prob6():
    """ Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """


    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = np.linspace(-2 * np.pi, 2 * np.pi, 100)

    X, Y = np.meshgrid(x, y)

    g = (np.sin(X) * np.sin(Y)) / (X * Y)

    plt.subplot(122)


    plt.subplot(121)
    plt.pcolormesh(X, Y, g, cmap="viridis", shading="auto")
    plt.colorbar()
    plt.xlim([-2 * np.pi, 2 * np.pi])
    

    plt.subplot(122)
    plt.contour(X, Y, g, 10, cmap="coolwarm")
    plt.colorbar()
    plt.ylim([-2 * np.pi, 2 * np.pi])

    plt.tight_layout()
    plt.show()


#testing

prob6()