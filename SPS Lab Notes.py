import numpy as np
from scipy import stats
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import data, io, color, transform, exposure

# SETTING PLT RENDERING VALUES
# By default we set figures to be 6"x4" on a 110 dots per inch (DPI) screen 
# (adjust DPI if you have a high res screen!)
plt.rc('figure', figsize=(6, 4), dpi=110)
plt.rc('font', size=10)


#########################################
############# INPUT OUTPUT ##############
#########################################


# LOADING DATA
D = np.loadtxt("data.dat", delimiter=",")
print(D.shape)

# SEPERATING CSV INTO TWO VARIABLES
x = D[:,0]
y = D[:,1]

# SAVING DATA
np.savetxt("dataOutput.dat", R, delimiter=',')


#########################################
########### PLOTTING DATA ###############
#########################################


# PLOTTING A SCATTER GRAPH
fig, ax = plt.subplots()
ax.scatter(x, y) #where x and y are arrays of points.
plt.show()

# LABELLING AXES
fig, ax = plt.subplots()
ax.set_xlabel("Labelling the X Axis")
ax.set_ylabel("Labelling the Y Axis")
ax.scatter(x,y)
plt.show()

# PLOTTING A 3D SCATTER
fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
x = D[:,1]
y = D[:,2]
z = D[:,3]
ax.scatter(x, y, z)
plt.show()

# PLOTTING HISTOGRAMS
fig, ax = plt.subplots()
plt.hist(x)
plt.show()

# - Specifying Bin Count and Range
R = np.random.randn(1000)
fig, ax = plt.subplots()
plt.hist(R, bins = 100, range = (-5, 5))
plt.show()


###################################################
############## MATRIX MATHEMATICS #################
###################################################


# DEFINING MATRICES
A = np.matrix('2 3; 4 -1; 5 6')
B = np.matrix([[5, 2],[8, 9],[2, 1]])

# FIND THE INVERSE OF A MATRIX

Ainv = np.linalg.inv(A)

# 2D NORMAL DISTRIBUTION PROBABILITY DENSITY FUNCTION

mu = np.array([2, 2])
cm = np.array([[4,2], [2, 6]])
x = (1, 2)

probability = stats.multivariate_normal.pdf(x, mean=mu, cov=cm)


###################################################
########## GENERATING RANDOM SEQUENCES ############
###################################################


# GENERATE A RANDOM SEQUENCE FROM THE NORMAL DISTRIBUTION
P = np.random.rand(100)

def customNormalDistribution(n=100, mean_dist=0, var_dist=0):
    sequence = np.sqrt(var_dist) * np.random.randn(n) + mean_dist
    return sequence

# GENERATE RANDOM VECTORS FROM NORMAL DISTRIBUTION

mu = np.array([2, 2])
cm = np.array([[4,2], [2, 6]])
vectors = stats.multivariate_normal.rvs(mean=mu, cov=cm, size=100) #size is number of vectors


###############################################
############ IMAGE MANIPULATION ###############
###############################################

# Make sure you have
from skimage import data, io, color, transform, exposure
# at the top of your script.

im_flower = io.imread('flower.png')
io.imshow(im_flower)
print('Image shape:', im_flower.shape)
im_flower_gray = color.rgb2gray(im_flower)

def imhist(img_hist):
    fig, ax = plt.subplots()
    ax.bar( range(256), img_hist[0], width=1 )
    ax.set_xlim(0, 256)

# Histogram of image 
im_flower_hist = exposure.histogram(im_flower_gray_half)
imhist(im_flower_hist)

# Equalising Exposure
im_eq = exposure.equalize_hist(im_flower_gray_half)
io.imshow(im_eq)

im_eq_hist = exposure.histogram(im_eq)
imhist(im_eq_hist)

