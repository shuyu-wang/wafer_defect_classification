import numpy as np
import keras
from keras import regularizers, metrics
from keras.layers import Input, Conv2D, MaxPool2D, Dense, BatchNormalization, add, GlobalAvgPool2D
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K


class VAEDLM():
    """
    Wafer map classification model

    """
    def __init__(self, dim, nclass, faulty_case):
        """
        Initialization

        :param dim: Wafer map's dimension
        :param nclass: Number of classes
        :param faulty_case: A list of data faulty case
        """
        self.dim = dim
        self.nclass = nclass
        self.faulty_case = faulty_case

    def conv2d_bn(self, x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        """
        conv2d -> batch normalization -> relu activation

        :param x: An unit includes a convolution layer, a batchnorm and activation
        :param nb_filter: Number of filter
        :param kernel_size:
        :param strides:
        :param padding:
        :return:
        """
        x = Conv2D(nb_filter, kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)

        x = keras.layers.LeakyReLU(alpha=0.01)(x)  # leaky relu
        return x

    def shortcut(self, input, residual):
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_height = int(round(input_shape[1] / residual_shape[1]))
        stride_width = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        identity = input
        # Adjust by 1x1 convolution if needed
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            identity = Conv2D(filters=residual_shape[3],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_regularizer=regularizers.l2(0.0001))(input)

        return add([identity, residual])

    def basic_unit(self, nb_filter, strides=(1, 1)):
        """
        Basic unit of block

        :param nb_filter: Number of filter
        :param strides:
        :return: An unit with shortcut connection
        """
        def f(input):
            conv1 = self.conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
            residual = self.conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))

            return self.shortcut(input, residual)

        return f

    def basic_block(self, nb_filter, repetitions, is_first_layer=False):
        """
        Basic block of model

        :param nb_filter: Number of filter
        :param repetitions: Times of repetition
        :param is_first_layer:
        :return:
        """

        def f(input):
            for i in range(repetitions):
                strides = (1, 1)
                if i == 0 and not is_first_layer:
                    strides = (2, 2)
                input = self.basic_unit(nb_filter, strides)(input)
            return input

        return f

    def my_model(self):
        input_shape = (self.dim, self.dim, 3)
        nclass = self.nclass

        input_ = Input(shape=input_shape)

        conv1 = self.conv2d_bn(input_, 64, kernel_size=(3, 3), strides=(2, 2))
        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        conv2 = self.basic_block(64, 2, is_first_layer=False)(pool1)
        pool2 = GlobalAvgPool2D()(conv2)

        output_ = Dense(nclass, activation='softmax')(pool2)

        model = Model(inputs=input_, outputs=output_)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adagrad",
                      metrics=["accuracy",
                               metrics.AUC(),
                               metrics.Precision(),
                               metrics.Recall()
                               ])
        return model

    def training(self,
                 data_for_train,
                 label_for_train,
                 callback_monitor='val_loss',
                 callback_min_delta=0.002,
                 callback_patiency=30,
                 validation_split=0.3,
                 shuffle=True,
                 epochs=200,
                 batch_size=1024,
                 verbose=2
                 ):
        self.model = self.my_model()

        callback = keras.callbacks.EarlyStopping(monitor=callback_monitor,
                                                 min_delta=callback_min_delta,
                                                 patience=callback_patiency)
        history = self.model.fit(data_for_train, label_for_train,
                                 validation_split=validation_split,
                                 shuffle=shuffle,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 callbacks=[callback]
                                 )
        return history

    def predict(self, x, y):
        return self.model.evaluate(x=x, y=y, verbose=2)

    def classification_result(self, x, y):
        for i in range(9):
            y_test = y[y[:, i] == 1]
            x_test = x[y[:, i] == 1]
            print(self.faulty_case[i], self.predict(x_test, y_test))
        print('-----------------------------------')
        print("Average score: {}".format(self.predict(x, y)))
        print('-----------------------------------')


if __name__ == '__main__':
    dim = 26
    nclass = 9
    faulty_case = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc',
                   'Near-full', 'Random', 'Scratch', 'none']

    wm_path = r"your_file_path"
    wm_label_path = r"your_file_path"

    wm_bin = np.load(wm_path)
    wm_label = np.load(wm_label_path)

    # make string label data to numerical data
    for i, l in enumerate(faulty_case):
        wm_label[wm_label == l] = i

    wm_label_onehot = to_categorical(wm_label)

    wm_train, wm_test, label_train, label_test = train_test_split(wm_bin,
                                                                  wm_label_onehot,
                                                                  test_size=0.2,
                                                                  random_state=76)

    model = VAEDLM(dim=dim, nclass=nclass, faulty_case=faulty_case)
    model.training(data_for_train=wm_train, label_for_train=label_train)
    model.classification_result(wm_test, label_test)
