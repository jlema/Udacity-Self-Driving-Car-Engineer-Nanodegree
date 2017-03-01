import argparse
import json
import pandas as pd
import numpy as np

from image_preproc import load_image, reduce_image, flip_image, trans_image, crop_image, scale_brightness_image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

ch, row, col = 3, 36, 160  # resized camera format

def get_model():

  model = Sequential()
  # Normalize data to -0.5 to 0.5 range
  model.add(Lambda(lambda x: x/127.5 - 1.,
           input_shape=(row, col, ch),
           output_shape=(row, col, ch)))
  # The next layer is 3 1X1 filters, this has the effect of transforming the color space of the images.
  # As we do not know the best color space beforehand, using 3 1X1 filters allows the model to choose its best color space
  model.add(Convolution2D(3, 1, 1, border_mode="same"))
  model.add(ELU())
  # We follow the NVIDIA architecture and create 3 convolutional layers with 2x2 stride and 5x5 kernel
  model.add(Convolution2D(3, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  # We follow the NVIDIA architecture and create 2 convolutional layers with no stride and 3x3 kernel
  model.add(Convolution2D(48, 3, 3, border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, border_mode="same"))
  model.add(Flatten())
  model.add(ELU())
  # We follow the NVIDIA architecture and create 3 fully connected layers
  model.add(Dense(100))
  model.add(ELU())
  model.add(Dense(50))
  model.add(ELU())
  model.add(Dense(10))
  model.add(ELU())
  model.add(Dense(1))
  # We optimize using Adam optimizer for Mean Square Error
  model.compile(optimizer="adam", loss="mse")

  return model

# training generator    
def gen(image_names, steering, batch_size, augmentate=True):
    while True:
        # get a random sample of images of size batch size without replacement
        batch_mask = np.random.choice(image_names.index, size=batch_size, replace=False)
        x = []
        y = []
        image_path = ''

        for i in range(batch_size):
            index = batch_mask[i]
            # load original steering angle
            steer = steering[index]
            # randomly remove lower steering angles (< 0.1)
            if abs(steer) < 0.1:
                if np.random.randint(2) == 0:
                    continue
            # if we are augmentating (i.e. generating training data)
            if (augmentate):
                # randomly choose left, center or right images
                # and apply a small shift to the steering angle to compensate
                rand = np.random.randint(3)
                if (rand == 0):
                    image_path = data['left'][index]
                    comp = .25
                if (rand == 1):
                    image_path = data['center'][index]
                    comp = 0.
                if (rand == 2):
                    image_path = data['right'][index]
                    comp = -.25
                steer = steer + comp
                image = load_image(image_path)
                # cut off unnecessary top and bottom parts of image
                image = crop_image(image)
                # translate images horizontally and vertically
                image, steer = trans_image(image, steer)
                # increase/decrease brightness
                image = scale_brightness_image(image)
                # reduce size of image
                image = reduce_image(image)
                # flip images and steering angles
                image, steer = flip_image(image, steer)
            # if we are NOT augmentating (i.e. generating validation data)
            else:
                # load original image
                image_path = data['center'][index]
                image = load_image(image_path)
                # cut off unnecessary top and bottom parts of image
                image = crop_image(image)
                # reduce size of image
                image = reduce_image(image)
            x.append(image)
            y.append(steer)
        x = np.array(x)
        y = np.array(y)

        yield x, y

if __name__ == "__main__":

  # parse arguments
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--batch', type=int, default=287, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=19803, help='How many frames per epoch.')
  args = parser.parse_args()

  # import driving log
  data = pd.read_csv('driving_log.csv') 
  # split data into training and validation
  X_train, X_val, y_train, y_val = train_test_split(data['center'], data['steering'])
 
  model = get_model()
  model.summary()
  model.fit_generator(
    gen(X_train, y_train, args.batch),
    samples_per_epoch=args.epochsize,
    nb_epoch=args.epoch,
    validation_data=gen(X_val, y_val, args.batch, False), # do not augmentate validation samples
    nb_val_samples=len(X_val),
    callbacks = [TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)]
  )
  
  print("Saving model weights and configuration file.")
  model.save_weights("model.h5", True)
  with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)