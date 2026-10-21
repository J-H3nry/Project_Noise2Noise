class SGD():
    def __init__(self, model_parameters, lr):  # model_parameters le fichier parameters de sequential
        self.model_parameters = model_parameters
        self.lr = lr

    def zero_grad(self):
        for i in range(len(self.model_parameters)):
            if not self.model_parameters[i]:
                self.model_parameters[i][0][1] *= 0
                self.model_parameters[i][1][1] *= 0

    def step(self):
        for i in range(len(self.model_parameters)):
            if not self.model_parameters[i]:
                self.model_parameters[i][0][0] -= self.lr*self.model_parameters[i][0][1]
                self.model_parameters[i][1][0] -= self.lr*self.model_parameters[i][1][1]
