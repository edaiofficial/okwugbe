import numpy as np


class EarlyStopping:
    def __init__(self, patience, path, best_score, min_loss, verbose=True, delta=0, trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = min_loss
        self.delta = delta
        self.trace_func = trace_func
        self.path = path

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation WER decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}) - Saving best model to {self.path}')
        self.val_loss_min = val_loss
