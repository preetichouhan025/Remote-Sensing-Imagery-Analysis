# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

class EarlyStopping():
    '''provides early stopping of training based on train and val loss difference(min_delta) and number of steps(tolerance)'''

    def __init__(self, tolerance=3, min_delta=0.01):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (float(train_loss) - float(validation_loss)) < self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True