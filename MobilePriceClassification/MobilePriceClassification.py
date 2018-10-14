import sys, csv
import numpy as np
import keras
from keras import models
from keras import layers
# import keras.callbacks.ModelCheckpoint
# import keras.callbacks.EarlyStopping
from keras.utils.np_utils import to_categorical

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



train_data =  np.array(Train_Data_List);
tensort_train_targets = np.array(Train_Target_List);
train_targets = to_categorical(tensort_train_targets)
test_data =  np.array(Test_Data_List);

print(train_data)
print(test_data)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

num_epochs = 90

model = build_model()
model.fit(train_data, train_targets,epochs=num_epochs, batch_size=4, verbose=0)
#model.save('mpc.h5')

# predictor = build_model()
# predictor.load_weights('mpc.h5')

predictions = model.predict(test_data)

with open('result.csv', 'w') as f:
    fieldnames = ['id', 'price_range']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1000):
        prediction = np.argmax(predictions[i])
        writer.writerow({'id': i+1, 'price_range': prediction})




# def onehot_to_label(onehot):
#     for i,code in enumerate(onehot):
#         if code == 1:
#             return i
#     return 4

#print(np.argmax(predictions[10]))
# val_categorical_crossentropy, val_accuracy = modeltest.evaluate(train_data, train_targets, verbose=0)
# print(val_accuracy)
# callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc',patience=1,),keras.callbacks.ModelCheckpoint(filepath='mpc.h5',monitor='val_loss',save_best_only=True,)]

# k = 4
# num_val_samples = len(train_data) // k
#
#
# all_accuracy_histories = []
#
# for i in range(k):
# 	print('processing fold #', i)
# 	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
# 	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
# 	partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
# 	partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
# 	model = build_model()
# 	model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=4, verbose=0)
# 	val_categorical_crossentropy, val_accuracy = model.evaluate(val_data, val_targets, verbose=0)
# 	all_accuracy_histories.append(val_accuracy)
#
# def mean(numbers):
#     return float(sum(numbers)) / max(len(numbers), 1)
#
# print(mean(all_accuracy_histories))
