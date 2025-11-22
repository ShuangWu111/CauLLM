import torch
import torch.nn as nn
from torch.nn import functional as F
from models.cdw_cross_entropy_loss import CDW_CELoss

def get_loss_module(config):

    task = config['task']
    mode = config['mode']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        if mode == 'ours':
            return SumLoss(reduction='none')
        else:
            return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample
        

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""
    sum_loss = 0
    for name, param in model.named_parameters():
        if 'ln' in name or 'wpe' in name:
            sum_loss += torch.sum(torch.square(param))
            break
    return sum_loss


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
    

class SumLoss(nn.CrossEntropyLoss):
    def forward(self,inp,target):
        device = torch.device('cuda:{}'.format(0))
        prediction = F.softmax(inp, dim=1)
        target_squeeze = target.squeeze(1)
        #one_hot_target = F.one_hot(target_squeeze, num_classes=5).double()
        loss_cdw = CDW_CELoss(num_classes = 5, 
                 alpha = 2, # Weight or penalty term
                 reduction  = "none",
                 transform  = "log",
                 eps = 1e-8).to(device) # Original paper uses power transform
                 
        l_cdw = loss_cdw(prediction, target_squeeze)
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction) + l_cdw

class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
