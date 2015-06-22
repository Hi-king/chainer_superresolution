import chainer

PATCH_SHAPE = (5, 5, 3)

class Model(chainer.FunctionSet):
    def __init__(self):
        super(Model, self).__init__(
            conv1 = chainer.functions.Convolution2D(
                in_channels=3,
                out_channels=64,
                ksize=3,
                pad=1),
            conv2 = chainer.functions.Convolution2D(
                in_channels=64,
                out_channels=128,
                ksize=PATCH_SHAPE[0],
                pad=0),
            fc1 = chainer.functions.Linear(128, 64),
            fc2 = chainer.functions.Linear(64, 3)
        )

    def train(self, x_data, y_data, optimizer):
        #x = chainer.Variable(x_data.reshape(1, PATCH_SHAPE[0]*PATCH_SHAPE[1]*3))
        y = chainer.Variable(y_data.reshape(y_data.size/3, 3))
        h = self.predict(x_data)
        optimizer.zero_grads()

        error = chainer.functions.mean_squared_error(h, y)
        error.backward()
        optimizer.update()
        print(error.data)

    def predict(self, x_data):
        print x_data.shape
        x = chainer.Variable(x_data)        
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
