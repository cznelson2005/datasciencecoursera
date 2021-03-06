import os
os.chdir('D:\\Udemy\\P16-Deep-Learning-AZ\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 1 - Artificial Neural Networks (ANN)\\Section 2 - Part 1 - ANN\\Artificial_Neural_Networks')
import keras
import pandas as pd
import numpy as np

# part 1- Data preprocessing

#Importing dataset
df_bank=pd.read_csv('Churn_Modelling.csv')
X=df_bank.iloc[:,3:13].values
y=df_bank.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part 2- make ANN
from keras.models import Sequential #to initialize ANN
from keras.layers import Dense #to create layers in ANN
from keras.layers import Dropout

#Initializing the ANN
classifier=Sequential()

#Ading the first layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
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
classifier.fit(X_train, y_train, nb_epoch=100, batch_size=10)

#predict the test set results
y_pred=classifier.predict(X_test)
y_pred = (y_pred > 0.5) #larger than 0.5 as True

# part 3- make prediction and evaluate the model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Homework
#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000

new_predictor=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_predictor=(new_predictor>0.5)

#Part 4: model evaluation

#k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier #wrap to combine cross validation into sklearn
from sklearn.model_selection import cross_val_score #import cross validation
from keras.models import Sequential #to initialize ANN
from keras.layers import Dense #to create layers in ANN
from keras.layers import Dropout #randomly disable some neurons to avoid overfitting

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))#try 10% of neurons being dropped
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu')) #input dim in unnecessary
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) #not using softmax here: applied to more than two categories
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #adam: gradient descent method, loss function
    return classifier

classifier=KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean=accuracies.mean()
variance=accuracies.std() #check variance, low variance is good

# Model tunning

#Tune grid
from keras.wrappers.scikit_learn import KerasClassifier #wrap to combine cross validation into sklearn
from sklearn.model_selection import GridSearchCV #import cross validation
from keras.models import Sequential #to initialize ANN
from keras.layers import Dense #to create layers in ANN
from keras.layers import Dropout #randomly disable some neurons to avoid overfitting

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier)

#grid search code
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters= grid_search.best_params_
best_accuracy= grid_search.best_score_

#prediction
Y_pred_grid=grid_search.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,Y_pred_grid)

#############################
#Convolutional Neural Network
#############################
import os
os.chdir('D:\\Udemy\\P16-Deep-Learning-AZ\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)')

#part 1: Building CNN
#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing the CNN
classifier=Sequential()

#step 1- convolution
#32 feature detectors, each one 3X3
#input_shape (dimension, dimension,channel) (Tensorflow backend format): the shape of input images, 
#all images need to have the same shape
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))

#step 2- pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

# adding a second convolution layer and pooling layer
classifier.add(Convolution2D(64,3,3, activation='relu')) #keras already knows the shape from previous pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

#step 3- flatten
classifier.add(Flatten())

#step 4- full connection
#add layers: the number of nodes not too big not too small
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid')) #output node: either dog or cat between 0 and 1

#compiling the CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# part 2- Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) #to have value between 0 and 1

training_set = train_datagen.flow_from_directory('Dataset/training_set',
                                                target_size=(64, 64), #size in CNN
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('Dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)


#part 3- make new prediction
import numpy as np
from keras.preprocessing.image import image
test_image=image.load_img('Dataset/single_prediction/cat_or_dog_3.jpg',target_size=(64, 64))#load the image for prediction
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0) #convert the 3 dimension to 4 dimension as CNN's input request
result=classifier.predict(test_image) #make prediction
training_set.class_indices #cat and dog indices, result shows 1 so is dog
if result[0][0] == 1:
    print('dog')
else:
    print('cat')

# Save model
#model_backup_path = os.path.join(script_dir, '../dataset/cat_or_dogs_model.h5')
#classifier.save(model_backup_path)
#print("Model saved to", model_backup_path)
    

#########################
#Recurrent Neural Network
#########################
import os
os.chdir('D:\\Udemy\\P16-Deep-Learning-AZ\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 3 - Recurrent Neural Networks (RNN)')    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Part 1- Data Preprocessing

#Importing the training set
data_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=data_train.iloc[:,1:2].values #numpy array, only take the opening price, set to 2 will become a numpy array of one column instead of a series

#Feature scaling
from sklearn.preprocessing import MinMaxScaler #Min max normalization
sc=MinMaxScaler(feature_range=(0,1))
training_set_scale=sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
#RNN will check 60 previous time steps and predict t+1 output
X_train=[]
y_train=[]

for i in range(60,len(training_set_scale)):
    X_train.append(training_set_scale[i-60:i,0]) #get the previous 60 prices
    y_train.append(training_set_scale[i,0]) #get t+1 value

X_train, y_train=np.array(X_train), np.array(y_train)

#Reshaping
X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1)) #add new dimension, 2D to 3D (batch_size, timesteps, input_dim)

#Part 2- Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout #avoid overfitting

#Initializing the RNN
regressor=Sequential()

#Adding the first LSTM layer and some Dropout regularisation
# 50 neurons give model with high dimensionality
# Return sequence: since we need to add one more layer, thus True
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))#drop 20% of neurons
#Adding second layer of LSTM
regressor.add(LSTM(units=50, return_sequences=True)) #input shape is not required anymore, already stated earlier
regressor.add(Dropout(0.2))#drop 20% of neurons
#Adding third layer of LSTM
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))#drop 20% of neurons
#Adding fourth layer of LSTM
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))#drop 20% of neurons

#Adding output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')#regression, thus MSE

#Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

#Part 3- making the prediction and visualizing the results

#Getting the real stock price of 2017
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values #numpy array, only take the opening price, set to 2 will become a numpy array of one column instead of a series

#Getting the predicted stock price of 2017
dataset_total=pd.concat([data_train['Open'],dataset_test['Open']], axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1) #reshape array into 3D format for the fit method to use
inputs=sc.transform(inputs)  #only transform to use the same scale from train dataset

X_test=[]

for i in range(60,80):
    X_test.append(inputs[i-60:i,0]) #get the previous 60 prices

X_test=np.array(X_test)

#Reshaping
X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1)) #add new dimension, 2D to 3D (batch_size, timesteps, input_dim)

predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualizing the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()



###
###SOM: Self-organizing maps
###
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
    
#Finding the fraud
mappings=som.win_map(X) #create mapping: each node assigns to each winning node
frauds=np.concatenate((mappings[(8,1)], mappings[(6,8)]),axis=0) #coordinate of the winning nodes
frauds=sc.inverse_transform(frauds)

