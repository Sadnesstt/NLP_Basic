from copy import deepcopy

import numpy as numpy

import torch

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from simple_ntc.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func) # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign model as class variable, we can easily access.
        engine.optimizer.zero_grad()

        x, y = mini_batch.text, mini_batch.label
        x, y = x.to(engine.device), y.to(engine.device)

        x = x[:, :engine.config.max_length]

        # Take feed-forward
        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y)
        loss.backward()

        # Calculate accuracy only if 'y' is LongTensor.
        # which means that 'y' is one-hot representation.
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim = -1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Take a step of gradient descent.
        engine.optimizer.step()

        return {
            'loss' : float(loss),
            'accuracy' : float(accuracy),
            '|param|' : p_norm,
            '|g_param|' : g_norm,
        }