import chainer

class Model(object):
    def __init__(self):
        self.optimizer = chainer.optimizers.Adam()
        #self.optimizer = chainer.optimizers.SGD(lr=0.0000001)
        self.functions = Functions()
        self.optimizer.setup(self.functions)
        self.PATCH_SHAPE = (19, 19, 3)


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
        return self.functions.forward(x)[0].data

    def to_gpu(self):
        self.functions.to_gpu()
        self.optimizer.setup(self.functions.collect_parameters())

class Functions(chainer.FunctionSet):
    def __init__(self):
        super(Functions, self).__init__(
            conv1 = chainer.functions.Convolution2D(
                in_channels=3,
                out_channels=32,
                ksize=5,
                stride=1,
                pad=0),
            conv2 = chainer.functions.Convolution2D(
                in_channels=32,
                out_channels=32,
                ksize=5,
                stride=1,
                pad=0),
            conv3 = chainer.functions.Convolution2D(
                in_channels=32,
                out_channels=32,
                ksize=5,
                stride=1,
                pad=0),
            conv4 = chainer.functions.Convolution2D(
                in_channels=32,
                out_channels=64,
                ksize=5,
                stride=1,
                pad=0),
            conv5 = chainer.functions.Convolution2D(
                in_channels=64,
                out_channels=128,
                ksize=3,
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
        conv1 = chainer.functions.relu(self.conv1(x))
        conv2 = chainer.functions.relu(self.conv2(conv1))
        conv3 = chainer.functions.relu(self.conv3(conv2))
        conv4 = chainer.functions.relu(self.conv4(conv3))
        conv5 = chainer.functions.relu(self.conv5(conv4))
        fc1 = chainer.functions.relu(self.fc1(conv5))
        fc2 = self.fc2(fc1)
        return fc2, {
            "conv1": conv1,
            "conv2": conv2,
            "conv3": conv3,
            "conv4": conv4,
            "conv5": conv5,
            "fc1": fc1,
            "fc2": fc2
        }

