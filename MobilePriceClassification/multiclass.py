import numpy as np
from keras import models
from keras import layers

from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
num_words=10000)
print(train_labels)


'''
instructions
model.save('cats_and_dogs_small_1.h5')
model.load_weights('pre_trained_glove_model.h5')
model.predict(x)


hidden units(64):
64- 71.50 70.50 70.99
32- 73.00 69.99 70.50 68.50


batch(4):
2-   76.49
4-   75.50 73.99
8-   73.99 72.50 72.50
16-  71.00 69.00
32-  72.00 72.00 73.99 69.00 72.50
64-  68.50


epochs():
20-74.49
30-74.99
40-75.99
50-74.00
60-74.50
70-75.00
80-75.50
90-76.99 77.00 78.99
100-72.50
110-76.99
12-71.00
130-74.50
140-77.49
150-70.99 78.00
160-
170-
180-
190-







'''
