#import basic libraries
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

# set up to read in the image files and pair them with their correct labels
df = pd.read_csv('labels.csv')
df.head()

n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

width = 299
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
for i in tqdm(range(n)):
    X[i] = cv2.resize(cv2.imread('train/%s.jpg' % df['id'][i]), (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1

# visualisation with matplotlib
import random
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(12, 6))
for i in range(8):
    random_index = random.randint(0, n-1)
    plt.subplot(2, 4, i+1)
    plt.imshow(X[random_index][:,:,::-1])
    plt.title(num_to_class[y[random_index].argmax()])

# import the keras applications
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input

# define the get_features function to extract bottleneck features using pre-trained models
def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features

# using the above function to extract bottleneck features with each pre-trained model
inception_features = get_features(InceptionV3, X)
xception_features = get_features(Xception, X)
resnet_features = get_features(ResNet50, X)
inresv2_features = get_features(InceptionResNetV2, X)

# combine all bottleneck features into one for training
features = np.concatenate([inception_features, xception_features, resnet_features, inresv2_features], axis=-1)

# set up our model and train with the bottleneck features from above
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch_with_4_predictors.hdf5', verbose=1, save_best_only=True)
h = model.fit(features, y, batch_size=128, epochs=100, callbacks=[checkpointer], validation_split=0.1)

# set up to read the image files for the test set and pair them with their correct labels
df2 = pd.read_csv('sample_submission.csv')

n_test = len(df2)
X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    X_test[i] = cv2.resize(cv2.imread('test/%s.jpg' % df2['id'][i]), (width, width))

# using the get_features function to extract bottleneck features from the test set
inception_t_features = get_features(InceptionV3, X_test)
xception_t_features = get_features(Xception, X_test)
resnet_t_features = get_features(ResNet50, X_test)
inresv2_t_features = get_features(InceptionResNetV2, X_test)

# combine all bottleneck features into one for testing
features_test = np.concatenate([inception_t_features, xception_t_features, resnet_t_features, inresv2_t_features], axis=-1)

# make predictions with our trained model
y_pred = model.predict(features_test, batch_size=128)

# export predictions to a csv file
for b in breed:
    df2[b] = y_pred[:,class_to_num[b]]

df2.to_csv('predictions.csv', index=None)