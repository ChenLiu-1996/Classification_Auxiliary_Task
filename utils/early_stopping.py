### This is modified from Bjarten/early-stopping-pytorch
# (https://github.com/Bjarten/early-stopping-pytorch)

from os import error
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if the criterion doesn't improve after a given patience."""
    def __init__(
            self, mode='min',
            criterion_name='validation loss',
            patience=7, verbose=False, delta=0,
            path='checkpoint.pt', trace_func=print
        ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.mode = mode
        if self.mode not in ['min', 'max']:
            return error("mode must be either `min` or `max`.")
        self.criterion_name = criterion_name
        if self.mode == 'min':
            self.best_achieved = np.Inf
        elif self.mode == 'max':
            self.best_achieved = -np.Inf
        self.best_score = None
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, criterion_new_value, model):

        if self.mode == 'min':
            score = -criterion_new_value
        elif self.mode == 'max':
            score = criterion_new_value

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(criterion_new_value, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(criterion_new_value, model)
            self.counter = 0

    def save_checkpoint(self, criterion_new_value, model):
        '''Saves model when criterion improves.'''
        if self.verbose:
            self.trace_func(f'{self.criterion_name:s} improved: ({self.best_achieved:.6f} --> {criterion_new_value:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_achieved = criterion_new_value