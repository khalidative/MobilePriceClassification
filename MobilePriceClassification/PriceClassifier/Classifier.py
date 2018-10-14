#=======================================================================
#Using Keras with tensorflow backend
#=======================================================================
import sys, csv
import numpy as np
import keras
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical

#=======================================================================
#Import training and test data from the csv files
#=======================================================================

Train_Data_List = []
Train_Target_List = []
Test_Data_List = []

with	open('train.csv',	'r')	as	f:
				reader	=	csv.DictReader(f,	delimiter=',')
				for	row	in	reader:
								Battery_power	=	float(row["battery_power"])
								Blue	=	float(row["blue"])
								Clock_speed = float(row["clock_speed"])
								Dual_sim = float(row["dual_sim"])
								Fc = float(row["fc"])
								Four_g = float(row["four_g"])
								Int_memory = float(row["int_memory"])
								M_dep = float(row["m_dep"])
								Mobile_wt = float(row["mobile_wt"])
								N_cores = float(row["n_cores"])
								Pc = float(row["pc"])
								Px_height = float(row["px_height"])
								Px_width = float(row["px_width"])
								Ram = float(row["ram"])
								Sc_h = float(row["sc_h"])
								Sc_w = float(row["sc_w"])
								Talk_time = float(row["talk_time"])
								Three_g = float(row["three_g"])
								Touch_screen = float(row["touch_screen"])
								Wifi = float(row["wifi"])
								PriceRange	=	int(row["price_range"])
								Train_Data_List.append([Battery_power, Blue, Clock_speed,Dual_sim,Fc,Four_g,Int_memory,M_dep,Mobile_wt,N_cores
								,Pc,Px_height,Px_width,Ram,Sc_h,Sc_w,Talk_time,Three_g,Touch_screen,Wifi ])
								Train_Target_List.append(PriceRange)

with	open('test.csv',	'r')	as	f:
				reader	=	csv.DictReader(f,	delimiter=',')
				for	row	in	reader:
								Battery_power	=	float(row["battery_power"])
								Blue	=	float(row["blue"])
								Clock_speed = float(row["clock_speed"])
								Dual_sim = float(row["dual_sim"])
								Fc = float(row["fc"])
								Four_g = float(row["four_g"])
								Int_memory = float(row["int_memory"])
								M_dep = float(row["m_dep"])
								Mobile_wt = float(row["mobile_wt"])
								N_cores = float(row["n_cores"])
								Pc = float(row["pc"])
								Px_height = float(row["px_height"])
								Px_width = float(row["px_width"])
								Ram = float(row["ram"])
								Sc_h = float(row["sc_h"])
								Sc_w = float(row["sc_w"])
								Talk_time = float(row["talk_time"])
								Three_g = float(row["three_g"])
								Touch_screen = float(row["touch_screen"])
								Wifi = float(row["wifi"])
								Test_Data_List.append([Battery_power, Blue, Clock_speed,Dual_sim,Fc,Four_g,Int_memory,M_dep,Mobile_wt,N_cores
								,Pc,Px_height,Px_width,Ram,Sc_h,Sc_w,Talk_time,Three_g,Touch_screen,Wifi ])


#=======================================================================
#Convert the data into 2D Tensors
#=======================================================================

train_data =  np.array(Train_Data_List);
tensort_train_targets = np.array(Train_Target_List);
train_targets = to_categorical(tensort_train_targets)
test_data =  np.array(Test_Data_List);

#=======================================================================
#Printing tensors to the console
#=======================================================================

print(train_data)
print(test_data)

#=======================================================================
#Normalize the training and test data
#=======================================================================

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

#=======================================================================
#Deep learning model architecture
#=======================================================================

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#=======================================================================
#Training and saving the deep learning model
#=======================================================================

num_epochs = 90
model = build_model()
model.fit(train_data, train_targets,epochs=num_epochs, batch_size=32, verbose=0)
model.save('mpc.h5')

#=======================================================================
#Using the model for prediction
#=======================================================================

predictions = model.predict(test_data)

#=======================================================================
#Entering results into results.csv 
#=======================================================================

with open('result.csv', 'w') as f:
    fieldnames = ['id', 'price_range']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1000):
        prediction = np.argmax(predictions[i])
        writer.writerow({'id': i+1, 'price_range': prediction})
