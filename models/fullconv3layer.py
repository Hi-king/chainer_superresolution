import chainer

class Model(object):
    def __init__(self, PATCH_SHAPE):
        self.optimizer = chainer.optimizers.Adam()
        #self.optimizer = chainer.optimizers.SGD(lr=0.0000001)
        self.functions = Functions(PATCH_SHAPE)
        self.optimizer.setup(self.functions.collect_parameters())
        self.PATCH_SHAPE = PATCH_SHAPE


    def train(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data.reshape(y_data.size/3, 3, 1, 1))
        h = self.functions.forward(x)
        self.optimizer.zero_grads()
        error = chainer.functions.mean_squared_error(h, y)
        error.backward()
        self.optimizer.update()
        return error.data

    def predict(self, x_data):
        x = chainer.Variable(x_data)
        return self.functions.forward(x).data

    def to_gpu(self):
        self.functions.to_gpu()
        self.optimizer.setup(self.functions.collect_parameters())

class Functions(chainer.FunctionSet):
    def __init__(self, PATCH_SHAPE):
        super(Functions, self).__init__(
            conv1 = chainer.functions.Convolution2D(
                in_channels=3,
                out_channels=64,
                ksize=3,
                stride=1,
                pad=1),
            conv2 = chainer.functions.Convolution2D(
                in_channels=64,
                out_channels=128,
                ksize=PATCH_SHAPE[0],
                stride=1,
                pad=0),
            fc1 = chainer.functions.Convolution2D(
                in_channels=128,
                out_channels=64,
                ksize=1,
                stride=1,
                pad=0),
            fc2 = chainer.functions.Convolution2D(
                in_channels=64,
                out_channels=3,
                ksize=1,
                stride=1,
                pad=0),
        )
    def forward(self, x):
        h = self.fc2(
            chainer.functions.relu(
            self.fc1(
            chainer.functions.relu(
            self.conv2(
            chainer.functions.relu(
            self.conv1(
            x
        )))))))
        return h
