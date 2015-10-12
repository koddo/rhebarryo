import numpy as np
import random
import matplotlib.pyplot as plt # for drawing the example images
from sklearn.svm import SVC


def show_examples(data, row, col):
    
    fig, ax = plt.subplots(row,col)

    for i, a in enumerate(ax.flatten()):
         pic_number = random.randint(0, data.shape[0])      
         pic = np.array(data[pic_number,:]).reshape(28,28)
         a.imshow(pic, cmap = 'gray', interpolation = 'nearest')
         a.xaxis.set_visible(False)
         a.yaxis.set_visible(False)
 
    #uncomment if you want to see it at once
    #plt.show()

    #save the picture to the file
    plt.savefig('train_examples.png', dpi = 100)
    
     


print 'Reading training data...'

# using np.loadtxt for load train data with skipping first row because because of the format of the file
train = np.loadtxt("train.csv", dtype = np.float32, delimiter = ',', skiprows = 1)

# First column is labels, other colums are pixels
train_labels = train[:, 0]
train_data = train[:, 1:]

print ('Train set dim = {0}, label set dim = {1}'.format(train_data.shape,train_labels.shape))
    
print 'Print 25 random chosen pictures with labels...to the file'
print 'Open file with examples...'


#show several examples from training data. the result is saved to 'train_examples.png'-file 
# in the same
# row is the number of rows in subplot array
# col is the number of columns in subplot array
# 
row = 10
col = 10
show_examples(train_data, row, col)

#uncomment if you want to see it at once
plt.show()
    
