#!/usr/bin/env python3

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image


import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import os.path


DiscriminatorModelPath = 'discriminator_model.h5'
GeneratorModelPath = 'generator_model.h5'
CombinedModelPath = 'combined_model.h5'


class DCGan():
    def __init__(self):
        self.img_rows = 96
        self.img_cols = 96
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer_g = Adam(0.00002, 0.5)
        optimizer_d = Adam(0.00002, 0.5)

        self.load_image_file_lists()

        print('Loading images...')
        self.train_images = (np.array(list(map(self.load_image, self.image_files))) - 127.5) / 127.5
        print(self.train_images[0])
        print('Finish loading: ' + str(self.train_images.shape))

        if self.is_model_exists():
            print('Loading saved models...')
            self.discriminator = load_model(DiscriminatorModelPath)
            self.generator = load_model(GeneratorModelPath)
            self.combined = load_model(CombinedModelPath)
            print('Finish loading models.')
        else:
            print('No saved model found, creating new ones.')
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer_d, metrics = ['accuracy'])
            self.generator = self.build_generator()

            z = Input(shape = (self.latent_dim,))
            img = self.generator(z)
            self.discriminator.trainable = False
            validity = self.discriminator(img)
            self.combined = Model(z, validity)
            self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer_g)


    def is_model_exists(self):
        return     os.path.exists(DiscriminatorModelPath)\
               and os.path.exists(GeneratorModelPath)\
               and os.path.exists(CombinedModelPath)

    
    def build_generator(self):
        model = Sequential()

        model.add(Dense(1024 * 3 * 3, activation = 'relu', input_dim = self.latent_dim))
        model.add(Reshape((3, 3, 1024)))
        model.add(BatchNormalization(momentum = 0.8))
        
        model.add(Conv2DTranspose(1024, kernel_size = 5, padding = 'same', strides = (2, 2)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum = 0.8))

        model.add(Conv2DTranspose(512, kernel_size = 5, padding = 'same', strides = (2, 2)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum = 0.8))

        model.add(Conv2DTranspose(256, kernel_size = 5, padding = 'same', strides = (2, 2)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum = 0.8))

        model.add(Conv2DTranspose(128, kernel_size = 5, padding = 'same', strides = (2, 2)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum = 0.8))

        model.add(Conv2DTranspose(64, kernel_size = 5, padding = 'same', strides = (2, 2)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum = 0.8))
        
        model.add(Conv2DTranspose(self.channels, kernel_size = 5, padding = 'same'))
        model.add(Activation('tanh'))

        model.summary()
        noise = Input(shape = (self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    
    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, (5, 5), padding = 'same', input_shape = self.img_shape))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (5, 5), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (5, 5), padding = 'same', strides = (2, 2)))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(512, (5, 5), padding = 'same', strides = (2, 2)))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1024, activation = 'tanh'))
        model.add(Dense(1, activation = 'sigmoid'))
        
        model.summary()
        img = Input(shape = self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size = 128, sample_interval = 50):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, self.train_images.shape[0], batch_size)
            imgs = self.train_images[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.discriminator.save(DiscriminatorModelPath)
                self.generator.save(GeneratorModelPath)
                self.combined.save(CombinedModelPath)
                self.sample_images(epoch)
    

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])#, cmap = 'gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('images/%d.png' % epoch)
        plt.close()


    def load_image(self, path):
        return image.img_to_array(image.load_img(path))

    
    def load_image_file_lists(self):
        image_root_path = 'faces'
        self.image_files = list(map(lambda file : os.path.join(image_root_path, file), 
                                filter(lambda fileName : fileName.endswith('.jpg'), os.listdir(image_root_path))))


if __name__ == '__main__':
    dcgan = DCGan()
    dcgan.train(epochs = 100000, batch_size = 128, sample_interval = 200)
