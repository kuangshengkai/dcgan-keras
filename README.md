# dcgan-keras
A DCGAN based image generating implementation with Keras

This model processes training avatar files with size of 96x96 under the directory of './faces/*.jpg' and generates sample avatars every 200 epochs into './images/<epochs-number>.png'. The model implementation mainly refers to the paper of [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434) with several keypoints distinguishing to several other open-source version of DCGAN seen on github:
* Replace pooling layers with strided Conv2DTranspose (generator) and Conv2D (discriminator), allowing the network to learn its own spatial downsampling according to the paper;
* Use ReLU/LeakyReLU in generator/discriminator;
* Use batchnorm in both the generator and the discriminator.

Generated samples are shown below, trained from more than 50k comic avatar images and generated on epoch 70000 & 71000:

![epoch-70000](images/70000.png)

![epoch-71000](images/71000.png)

The raw training set is not provided in this repository, though you may replace it with your own images placing under the directory of './faces/*.jpg' with size of 96x96, or your own size with replacement of the ***img_rows*** & ***img_cols*** accordingly in the source code.
