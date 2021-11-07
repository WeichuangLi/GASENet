import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.Inf
        self.delta = delta

    def __call__(self, acc, loss, score_type, model, epoch):
        # if early stop is initialized for the 1st time
        if score_type == "acc":
            if self.best_score is None:
                self.best_score = acc
                self.save_checkpoint(acc, loss, model, epoch, score_type)

            # for accuracy, we hope this measurement becomes greater than best_score
            # (with a minimum of self.delta, which is negative)
            elif acc < self.best_score - self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            # if the measurement increase to a desired value, reset the counter
            else:
                self.best_score = acc
                self.save_checkpoint(acc, loss, model, epoch, score_type)
                self.counter = 0

        elif score_type == "loss":
            if self.best_score is None:
                self.best_score = loss
                self.save_checkpoint(acc, loss, model, epoch, score_type)

            # for accuracy, we hope this measurement becomes smaller than best_score
            # (with a minimum of self.delta, which is negative)
            elif loss > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            # if the measurement drop to a desired value, reset the counter
            else:
                self.best_score = loss
                self.save_checkpoint(acc, loss, model, epoch, score_type)
                self.counter = 0

    def save_checkpoint(self, val_acc, loss, model, epoch, score_type):
        """Saves model when measurement increase/decrease."""
        if score_type == "acc":
            if self.verbose:
                print(f'Validation accuracy increased ({self.score_max:.6f} --> {val_acc:.6f}).  Saving model ...')
            self.score_max = val_acc

        elif score_type == "loss":
            if self.verbose:
                print(f'Validation loss decreased ({self.score_max:.6f} --> {loss:.6f}).  Saving model ...')
            self.score_max = loss

        # Save model
        model.save_networks(epoch, val_acc)

