#Mega case study: Hybrid deep learning model


#Part 1: Identify the frauds with SOM

import os
os.chdir('D:\\Udemy\\P16-Deep-Learning-AZ\\Self_Organizing_Maps')    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Credit_Card_Applications.csv')

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))

X=sc.fit_transform(X)

#Train SOM: not under scikilearn, thus need to get from other developer
#Here use Minisom
from minisom import MiniSom
som=MiniSom(x=10, y=10, input_len=15,sigma=1.0, learning_rate=0.5) #create SOM object
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#Visualizing the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) #return the matrix of all the distances
colorbar() #legend of the colors, fraud nodes are far away from neighboring nodes, those 1.0

#add two markers, red circles do not get approval, green square gets approval
markers=['o','s'] #circle and square
colors=['r','g'] #red and green

for i, x, in enumerate(X): #loop through all the customer records and mark each winning node
    w=som.winner(x) #w as winning node, mark with color
    plot(w[0]+0.5, 
         w[1]+0.5, #coordinates x and y bottom left corner, thus +0.5 to put the marker at the center
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()
    
#Finding the fraud: the fraud is identified as label for supervised learning model
mappings=som.win_map(X) #create mapping: each node assigns to each winning node
frauds=np.concatenate((mappings[(1,4)], mappings[(7,3)]),axis=0) #coordinate of the winning nodes
frauds=sc.inverse_transform(frauds)

#Part 2: Going from unsupervising to supervised deep learning

#Create the matrix of features
customers=dataset.iloc[:,1:].values #include all columns except the first one

#Create the depedent variables
is_fraud=np.zeros(len(dataset)) #create vector then replace cheated customer with 1

for i in range(len(dataset)): #create y
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# part 2- make ANN
from keras.models import Sequential #to initialize ANN
from keras.layers import Dense #to create layers in ANN
from keras.layers import Dropout

#Initializing the ANN
classifier=Sequential()

#Ading the first layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=15))
classifier.add(Dropout(p=0.1))
#Adding the second layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu')) #input dim in unnecessary
classifier.add(Dropout(p=0.1))
#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) #softmax: applied to more than two categories

#Compiling the ANN
#Adam: the gradient descent method
#loss: loss function, here for binary we use binary_crossentropy
#metrics:measurement metrics
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #adam: gradient descent method, loss function

#Fitting the ANN to the training set
#batch_size: the size of the batch to train the weight
#epho: the number of times we train the ANN on the training set
classifier.fit(customers, is_fraud, nb_epoch=2, batch_size=2)

#predict the test set probabilities
y_pred=classifier.predict(customers)

y_pred=np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis=1) #concatenate customer id

y_pred=y_pred[y_pred[:,1].argsort()]
