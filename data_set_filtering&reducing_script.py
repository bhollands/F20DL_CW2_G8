#Written by Bernard Hollands 
import sys
assert sys.version_info>=(3,5)
import sklearn
assert sklearn.__version__>="0.20"

# common imports
import numpy as np 
import os
import cv2
import pandas as pd
#to make thos notebooks output stable across runs
np.random.seed(42)

#to plot pretty sigures
import matplotlib as mpl
import matplotlib.pyplot as plt

#define te file paths
here = os.path.dirname(os.path.abspath(__file__))
# x_train_smpl_bin_random  = os.path.join(here, 'x_train_smpl_bin_random.csv')
# x_train_smpl_bin_random_reduced = os.path.join(here, 'x_train_smpl_bin__random_reduced.csv')

x_train_gr_smpl_random = os.path.join(here, 'x_test_gr_smpl.csv')
x_train_gr_smpl_random_reduced = os.path.join(here, 'x_test_gr_smpl_reduced.csv')

#data = pd.read_csv(x_train_smpl_bin_random) #read in the data from the csv
data = pd.read_csv(x_train_gr_smpl_random) #read in the data from the csv
resize_length = 35 #define the new length length and width of the images

num_of_rows = 3091#9690#2431 #define the number of rows in the data file
num_of_colums_original = 2304 #define the number of columns in the data
num_of_colums_reduced = resize_length**2 #the number of colmns is the length squared

row = np.empty(num_of_colums_original) #row needs to be as long as the number of columbs

reduced_data_set = np.zeros(shape=(num_of_rows,num_of_colums_reduced)) #2D array to store entire set

for j in range(num_of_rows): #go through all the rows in the file
    print(j) #show the under progress throught the file
    for i in range(num_of_colums_original): #go through all of the colmns in the data
        row[i] = data[str(i)][j] #create an array with the data from each colmn of a specific row
    image = cv2.resize(row.reshape(48,48), dsize=(resize_length,resize_length)) #reduce the size of the image
    reduced_image_vector = image.reshape(1,num_of_colums_reduced) #make the reduced image a vector image agains
    reduced_data_set[j] = reduced_image_vector #write the reduced image to the 2d array


#np.savetxt(x_train_smpl_bin_random_reduced, reduced_data_set, delimiter=',', fmt='%d') #save the 2D array as a csv file
np.savetxt(x_train_gr_smpl_random_reduced, reduced_data_set, delimiter=',', fmt='%d')

