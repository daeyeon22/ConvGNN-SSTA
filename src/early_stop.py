

class EarlyStopping(object):
    def __init__(self, patience=5):
        self.n_patience = 5
        self.patience = 0
        self.history = []
        self.escape = False


    def update(self, val_loss):
        self.history.append(val_loss)



        





