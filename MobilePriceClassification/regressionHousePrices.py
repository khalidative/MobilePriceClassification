import sys, csv
import numpy as np
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
# import matplotlib.pyplot as plt
# from keras.datasets import boston_housing

Train_Data_List = []
Train_Target_List = []

with	open('train.txt',	'r')	as	f:
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


train_data =  np.array(Train_Data_List);
tensort_train_targets = np.array(Train_Target_List);
train_targets = to_categorical(tensort_train_targets)

#print(train_data.shape)
#print(train_targets.shape)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
# test_data -= mean
# test_data /= std

# print(train_data)
# print(train_targets)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print(build_model())

num_epochs = 90

model = build_model()
model.fit(train_data, train_targets,epochs=num_epochs, batch_size=4, verbose=0)
val_categorical_crossentropy, val_accuracy = model.evaluate(train_data, train_targets, verbose=0)
print(val_accuracy)

# k = 4
# num_val_samples = len(train_data) // k


# all_accuracy_histories = []

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

# average_accuracy_history = [np.mean([x[i] for x in all_accuracy_histories]) for i in range(num_epochs)]
#
# def smooth_curve(points, factor=0.9):
# 	smoothed_points = []
# 	for point in points:
# 		if smoothed_points:
# 			previous = smoothed_points[-1]
# 			smoothed_points.append(previous * factor + point * (1 - factor))
# 		else:
# 			smoothed_points.append(point)
# 	return smoothed_points
#
# smooth_accuracy_history = smooth_curve(average_accuracy_history[10:])
#
# print(smooth_accuracy_history)

# plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

#print(all_scores);
