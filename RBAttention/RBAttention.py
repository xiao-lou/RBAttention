from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, \
    ReLU, DepthwiseConv2D, multiply
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)

    x = BatchNormalization(axis=3, scale=False)(x)

    if (activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def Block(U, inp):

    W = U  # 64

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, activation=None,
                         padding='same')

    conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3, activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3, activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out  # 64




def ResBottleneckBlock(filter, length, inp):

    W = filter  # 64
    shortcut = Conv2D(int(W), kernel_size=1, strides=(1, 1), padding='same', activation=None)(inp)
    conv1 = Conv2D(int(length * W), kernel_size=1, strides=(1, 1), padding='same', activation=None)(inp)
    conv1 = ReLU(6.)(conv1)
    conv2 = DepthwiseConv2D(kernel_size=3, strides=(1, 1), padding='same', activation=None)(conv1)
    conv2 = ReLU(6.)(conv2)
    conv2 = Conv2D(int(W), kernel_size=1, strides=(1, 1), padding='same', activation=None)(conv2)
    out = add([shortcut, conv2])
    return out  # 64


def AttentionBlock(x, g, inter_channels):

    W_x = Conv2D(inter_channels, kernel_size=1, strides=(1, 1), padding='same')(x)
    W_x = BatchNormalization(axis=3, scale=False)(W_x)

    W_g = Conv2D(inter_channels, kernel_size=1, strides=(1, 1), padding='same')(g)
    W_g = BatchNormalization(axis=3, scale=False)(W_g)

    f = Activation('relu')(add([W_x, W_g]))

    psi_f = Conv2D(1, kernel_size=1, strides=(1, 1), padding='same')(f)  # 1
    psi_f = BatchNormalization(axis=3, scale=False)(psi_f)
    rate = Activation('sigmoid')(psi_f)

    att_out = multiply([x, rate])

    return att_out



def RBAttention(height, width, n_channels):
    '''
    MultiResUNet

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras nets] -- MultiResUNet nets
    '''

    inputs = Input((height, width, n_channels))

    block1 = Block(64, inputs)  # 64
    resBottleneck1 = ResBottleneckBlock(64, 4, block1)  # 64
    pool1 = MaxPooling2D(pool_size=(2, 2))(block1)  # 64


    block2 = Block(128, pool1)  # 128
    resBottleneck2 = ResBottleneckBlock(128, 3, block2)  # 128
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)  # 128


    block3 = Block(256, pool2)  # 256
    resBottleneck3 = ResBottleneckBlock(256, 2, block3)  # 256
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)  # 256


    block4 = Block(512, pool3)  # 512
    resBottleneck4 = ResBottleneckBlock(512, 1, block4)  # 512
    pool4 = MaxPooling2D(pool_size=(2, 2))(block4)  # 512


    block5 = Block(1024, pool4)   # 1024


    d5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(block5)  # 256
    attention_block4 = AttentionBlock(resBottleneck4, d5, 256)  # 256

    up6 = concatenate([attention_block4, d5], axis=3)  # 256+256=512
    block6 = Block(512, up6)  # 512


    d6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(block6)  # 128
    attention_block3 = AttentionBlock(resBottleneck3, d6, 128)   # 128

    up7 = concatenate([attention_block3, d6], axis=3)  # 128+128=256
    block7 = Block(256, up7)  # 256


    d7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(block7)  # 64
    attention_block2 = AttentionBlock(resBottleneck2, d7, 64)  # 64

    up8 = concatenate([attention_block2, d7], axis=3)  # 64 + 64 = 128
    block8 = Block(128, up8)  # 128

    d8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(block8)  # 32
    attention_block1 = AttentionBlock(resBottleneck1, d8, 32)  # 32

    up9 = concatenate([attention_block1, d8], axis=3)  # 32 + 32 = 64
    block9 = Block(64, up9)  # 64


    conv10 = conv2d_bn(block9, 1, 1, 1, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def main():
    model = RBAttention(128, 128, 3)
    print(model.summary())


if __name__ == '__main__':
    main()
