import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input


class VAE():
    """
    Variational Auto Encoder
    """
    def __init__(self, dim, threshold, wm, label, epoch=15, batch_size=1024):
        """
        Initialization.

        :param dim: Dimension
        :param threshold: Threshold of augmentation
        :param wm: Wafer Map
        :param label: Wafer Map's label
        :param epoch:
        :param batch_size:
        """
        self.dim = dim
        self.threshold = threshold
        self.wm = wm
        self.label = label
        self.faulty_case = np.unique(label)
        self.epoch = epoch
        self.batch_size = batch_size

        self.build()

    def build(self):
        """
        Build model

        """
        # Encoder
        input_shape = (self.dim, self.dim, 3)
        input_tensor = Input(input_shape)
        encode = layers.Conv2D(64, (3, 3),
                               padding='same',
                               activation='relu')(input_tensor)

        latent_vector = layers.MaxPool2D()(encode)

        # Decoder
        decode_layer_1 = layers.Conv2DTranspose(64, (3, 3),
                                                padding='same',
                                                activation='relu')
        decode_layer_2 = layers.UpSampling2D()
        output_tensor = layers.Conv2DTranspose(3, (3, 3),
                                               padding='same',
                                               activation='sigmoid')

        # Connect Decoder Layers
        decode = decode_layer_1(latent_vector)
        decode = decode_layer_2(decode)

        self.ae = models.Model(input_tensor, output_tensor(decode))
        self.ae.compile(optimizer='Adam', loss='mse')

        self.ae.fit(self.wm, self.wm,
                    batch_size=self.batch_size,
                    epochs=self.epoch,
                    verbose=2)

        self.encoder = models.Model(input_tensor, latent_vector)

        decoder_input = Input((int(self.dim / 2), int(self.dim / 2), 64))
        decode = decode_layer_1(decoder_input)
        decode = decode_layer_2(decode)

        self.decoder = models.Model(decoder_input, output_tensor(decode))

    def gen_data(self, wafer, label):
        # Encode wafer maps
        encoded_wm = self.encoder.predict(wafer)

        gen_wm = np.zeros((1, self.dim, self.dim, 3))

        if label != 'none':
            # Add Gaussian Noise
            for i in range((self.threshold // len(wafer)) + 1):
                noised_encoded_wm = encoded_wm + np.random.normal(loc=0,
                                                                  scale=0.05,
                                                                  size=(len(encoded_wm),
                                                                        int(self.dim / 2),
                                                                        int(self.dim / 2),
                                                                        64))
                noised_gen_x = self.decoder.predict(noised_encoded_wm)
                gen_wm = np.concatenate((gen_wm, noised_gen_x), axis=0)
        else:
            # Keep the 'none' pattern a bigger amount due to the 'none' pattern
            # Add Gaussian Noise
            for i in range((13500 // len(wafer))):
                noised_encoded_wm = encoded_wm + np.random.normal(loc=0,
                                                                  scale=0.05,
                                                                  size=(len(encoded_wm),
                                                                        int(self.dim / 2),
                                                                        int(self.dim / 2),
                                                                        64))
                noised_gen_x = self.decoder.predict(noised_encoded_wm)
                gen_wm = np.concatenate((gen_wm, noised_gen_x), axis=0)

        gen_label = np.full((len(gen_wm), 1), label)

        return gen_wm[1:], gen_label[1:]

    def aug(self):
        """
        Augmentation

        :return: augmented wafer map, augmented wafer map's label
        """
        aug_wm = np.zeros((1, self.dim, self.dim, 3))
        aug_label = np.array(list()).reshape((-1, 1))

        for f in self.faulty_case:
            gen_x, gen_y = self.gen_data(self.wm[np.where(self.label == f)[0]], f)

            aug_wm = np.concatenate((aug_wm, gen_x), axis=0)
            aug_label = np.concatenate((aug_label, gen_y))

        # re onehot
        aug_wm_unonehot = np.argmax(aug_wm, axis=3)  # With zero head#

        aug_wm = np.zeros((len(aug_wm_unonehot), self.dim, self.dim, 3))
        for w in range(len(aug_wm_unonehot)):  #
            for i in range(self.dim):
                for j in range(self.dim):
                    aug_wm[w, i, j, int(aug_wm_unonehot[w, i, j])] = 1  #

        return aug_wm[1:], aug_label


if __name__ == '__main__':
    df = pd.read_pickle(r"dataset_filepath")

    def find_dim(x):
        dim0 = np.size(x, axis=0)
        dim1 = np.size(x, axis=1)
        return dim0, dim1

    df['waferMapDim'] = df.waferMap.apply(find_dim)

    dim = 26
    """
    Among all dimension's data from wm811k,
    the 26*26 have most abundant data, 
    so we use dim = 26 to verify our method
    """

    sub_df = df.loc[df['waferMapDim'] == (dim, dim)]
    sub_wafer = sub_df['waferMap'].values

    sw = np.ones((1, dim, dim))
    label = list()

    for i in range(len(sub_df)):
        if len(sub_df.iloc[i, :]['failureType']) == 0:
            continue
        sw = np.concatenate((sw, sub_df.iloc[i, :]['waferMap'].reshape(1, dim, dim)))
        label.append(sub_df.iloc[i, :]['failureType'][0][0])

    x = sw[1:]
    y = np.array(label).reshape((-1, 1))

    print(x.shape, y.shape)
    x = x.reshape((-1, dim, dim, 1))

    onehot_x = np.zeros((len(x), dim, dim, 3))

    for w in range(len(x)):
        for i in range(dim):
            for j in range(dim):
                onehot_x[w, i, j, int(x[w, i, j])] = 1

    vae_model = VAE(dim=dim, threshold=2100, wm=onehot_x, label=y)

    aug_wm, aug_label = vae_model.aug()

    # Save data
    np.save("target_path", aug_wm)
    np.save("target_path", aug_label)
