from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing import image
from keras.datasets import mnist
import numpy as np

def get_img(img_path):
    img = image.load_img(img_path, target_size=(128, 128)).convert(mode="L")
    # print "img ",
    x = image.img_to_array(img)
    # print "x ",x.shape
    x = np.expand_dims(x, axis=0)
    # print "x ",x.shape
    return x

# this is the size of our encoded representations
encoding_dim = 8192  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# this is our input placeholder
input_img = Input(shape=(16384,))
# "encoded" is the encoded representation of the input
encoded = Dense(8192, activation='relu')(input_img)
encoded = Dense(4096, activation='relu')(encoded)
encoded = Dense(2048, activation='relu')(encoded)
encoded = Dense(512, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(2048, activation='sigmoid')(encoded)
decoded = Dense(4096, activation='sigmoid')(decoded)
decoded = Dense(8192, activation='sigmoid')(decoded)
decoded = Dense(16384, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from os import listdir
from PIL import Image as PImage

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    x = []
    for image in imagesList:
    	x.append(get_img(path + image))
        # img = PImage.open(path + image)
        # loadedImages.append(img)

    return x 

path1 = "/home/neha/CS671/Tut04/autoencoder/Ear/"
path2 = "/home/neha/CS671/Tut04/autoencoder/Iris/"
path3 = "/home/neha/CS671/Tut04/autoencoder/Knuckle/"
path4 = "/home/neha/CS671/Tut04/autoencoder/Palm/"

# your images in an array
imgs1 = loadImages(path1)
imgs2 = loadImages(path2)
imgs3 = loadImages(path3)
imgs4 = loadImages(path4)

# print (imgs1)

len1 = int(len(imgs1) * 0.8)
len2 = int(len(imgs2) * 0.8)
len3 = int(len(imgs3) * 0.8)
len4 = int(len(imgs4) * 0.8)
# print (len1)
train_img = np.array(imgs1[0:len1] + imgs2[0:len2] + imgs3[0:len3] + imgs4[0:len4])
test_img = np.array(imgs1[len1:] + imgs2[len2:] + imgs3[len3:] + imgs4[len4:])

# img = np.array(img)
# length = len(img)
# rows = int(length*0.8)
x_train = train_img
x_test = test_img

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))



# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
import matplotlib.pyplot as plt

n = 15  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i+400].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


