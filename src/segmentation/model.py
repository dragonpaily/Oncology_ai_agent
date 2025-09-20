# --- Custom Layers and Helper Functions ---
class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(InstanceNormalization, self).build(input_shape)
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

def build_training_model(input_shape=(128, 128, 128, 4), n_classes=4, base_filters=32, dropout_rate=0.2):
    inputs = Input(input_shape)
    def conv_block_with_residual(x, filters, stage, dropout_rate=0.2):
        shortcut = x
        x = Conv3D(filters, 3, padding='same', use_bias=False, name=f'conv{stage}_1')(x)
        x = InstanceNormalization(name=f'in{stage}_1')(x)
        x = LeakyReLU(alpha=0.2, name=f'leaky{stage}_1')(x)
        if dropout_rate > 0: x = Dropout(dropout_rate, name=f'drop{stage}_1')(x)
        x = Conv3D(filters, 3, padding='same', use_bias=False, name=f'conv{stage}_2')(x)
        x = InstanceNormalization(name=f'in{stage}_2')(x)
        if int(shortcut.shape[-1]) == filters: x = Add(name=f'residual{stage}')([x, shortcut])
        else:
            shortcut = Conv3D(filters, 1, padding='same', name=f'shortcut{stage}')(shortcut)
            x = Add(name=f'residual{stage}')([x, shortcut])
        x = LeakyReLU(alpha=0.2, name=f'leaky{stage}_out')(x)
        if dropout_rate > 0: x = Dropout(dropout_rate/2, name=f'drop{stage}_out')(x)
        return x
    c1 = conv_block_with_residual(inputs, base_filters, '1', dropout_rate)
    p1 = MaxPooling3D(2, name='pool1')(c1)
    c2 = conv_block_with_residual(p1, base_filters * 2, '2', dropout_rate)
    p2 = MaxPooling3D(2, name='pool2')(c2)
    c3 = conv_block_with_residual(p2, base_filters * 4, '3', dropout_rate)
    p3 = MaxPooling3D(2, name='pool3')(c3)
    c4 = conv_block_with_residual(p3, base_filters * 8, '4', dropout_rate)
    p4 = MaxPooling3D(2, name='pool4')(c4)
    c5 = conv_block_with_residual(p4, base_filters * 16, '5', dropout_rate)
    u4 = UpSampling3D(2, name='up4')(c5)
    u4 = Conv3D(base_filters * 8, 2, padding='same', name='upconv4')(u4)
    att4 = tf.keras.layers.Multiply()([u4, c4])
    concat4 = Concatenate(name='concat4')([u4, att4])
    c6 = conv_block_with_residual(concat4, base_filters * 8, '6', dropout_rate)
    out4 = Conv3D(n_classes, 1, activation='softmax', name='out4')(c6)
    u3 = UpSampling3D(2, name='up3')(c6)
    u3 = Conv3D(base_filters * 4, 2, padding='same', name='upconv3')(u3)
    att3 = tf.keras.layers.Multiply()([u3, c3])
    concat3 = Concatenate(name='concat3')([u3, att3])
    c7 = conv_block_with_residual(concat3, base_filters * 4, '7', dropout_rate)
    out3 = Conv3D(n_classes, 1, activation='softmax', name='out3')(c7)
    u2 = UpSampling3D(2, name='up2')(c7)
    u2 = Conv3D(base_filters * 2, 2, padding='same', name='upconv2')(u2)
    att2 = tf.keras.layers.Multiply()([u2, c2])
    concat2 = Concatenate(name='concat2')([u2, att2])
    c8 = conv_block_with_residual(concat2, base_filters * 2, '8', dropout_rate)
    out2 = Conv3D(n_classes, 1, activation='softmax', name='out2')(c8)
    u1 = UpSampling3D(2, name='up1')(c8)
    u1 = Conv3D(base_filters, 2, padding='same', name='upconv1')(u1)
    att1 = tf.keras.layers.Multiply()([u1, c1])
    concat1 = Concatenate(name='concat1')([u1, att1])
    c9 = conv_block_with_residual(concat1, base_filters, '9', dropout_rate/2)
    outputs = Conv3D(n_classes, 1, activation='softmax', name='out_final')(c9)
    return Model(inputs=inputs, outputs=[outputs, out2, out3, out4])