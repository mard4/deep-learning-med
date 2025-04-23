import torch
import monai

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change to qualify as improvement.
            verbose (bool): Print when early stopping is triggered.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss  # because lower loss is better
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def create_loss(config):
    if config["loss"] == "Dice":
        return monai.losses.DiceLoss(sigmoid=True, batch=True)
    elif config["loss"] == "DiceCEL":
        return monai.losses.DiceCELoss(sigmoid=True, to_onehot_y=False, softmax=False)
    elif config["loss"] == "Focal":
        return monai.losses.FocalLoss()
    elif config["loss"] == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif config["loss"] == "CE":
        return torch.nn.CrossEntropyLoss()
    return None

def create_optimizer(config, model):
    if config["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    return None