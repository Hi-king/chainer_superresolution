import chainer

PATCH_SHAPE = (5, 5, 3)

class Model(chainer.FunctionSet):
    def __init__(self):
        super(Model, self).__init__(
            fc1 = chainer.functions.Linear(PATCH_SHAPE[0]*PATCH_SHAPE[1]*3, 128),
            fc2 = chainer.functions.Linear(128, 64),
            fc3 = chainer.functions.Linear(64, 3)
        )

    def train(self, x_data, y_data, optimizer):
        #x = chainer.Variable(x_data.reshape(1, PATCH_SHAPE[0]*PATCH_SHAPE[1]*3))
        y = chainer.Variable(y_data.reshape(y_data.size/3, 3))
        h = self.predict(x_data)
        #print(x.data.shape)
        #print(y.data.shape)
        optimizer.zero_grads()

        #print chainer.functions.mean_squared_error(
        #    chainer.Variable(originals),
        #    y
        #).data

        error = chainer.functions.mean_squared_error(h, y)
        error.backward()
        optimizer.update()
        print(error.data)

    def predict(self, x_data):
        x = chainer.Variable(x_data.reshape(x_data.size/PATCH_SHAPE[0]/PATCH_SHAPE[1]/3, PATCH_SHAPE[0]*PATCH_SHAPE[1]*3))        
        # h = self.fc2(
        #     chainer.functions.relu(
        #         self.fc1(x)
        #     ))
        h = self.fc3(
            chainer.functions.relu(
                self.fc2(
                    self.fc1(x)
            )))
        return h
