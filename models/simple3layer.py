import chainer

class Model(object):
    def __init__(self, PATCH_SHAPE):
        self.optimizer = chainer.optimizers.Adam()
        self.functions = Functions(PATCH_SHAPE)
        self.optimizer.setup(self.functions.collect_parameters())
        self.PATCH_SHAPE = PATCH_SHAPE

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
    def __init__(self, PATCH_SHAPE):
        super(Functions, self).__init__(
            fc1 = chainer.functions.Linear(PATCH_SHAPE[0]*PATCH_SHAPE[1]*3, 128),
            fc2 = chainer.functions.Linear(128, 64),
            fc3 = chainer.functions.Linear(64, 3)
        )
    def forward(self, x):
        h = self.fc3(
            chainer.functions.relu(
            self.fc2(
            self.fc1(x)
        )))
        return h
