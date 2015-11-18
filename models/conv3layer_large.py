import chainer

class Model(object):
    PATCH_SHAPE = (13, 13, 3)

    def __init__(self):
        self.optimizer = chainer.optimizers.Adam()
        self.functions = Functions()
        self.optimizer.setup(self.functions.collect_parameters())

    def train(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data.reshape(y_data.size/3, 3))
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
    def __init__(self):
        super(Functions, self).__init__(
            conv1 = chainer.functions.Convolution2D(
                in_channels=3,
                out_channels=64,
                ksize=9,
                stride=1,
                pad=0),
            conv2 = chainer.functions.Convolution2D(
                in_channels=64,
                out_channels=32,
                ksize=5,
                stride=1,
                pad=0),
            fc1 = chainer.functions.Linear(32, 16),
            fc2 = chainer.functions.Linear(16, 3)
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
