
import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing_coeff, ignore_index, rank='cuda:0'):
        assert 0.0 <= smoothing_coeff <= 1.0
        super().__init__()
        self.smoothing_coeff = smoothing_coeff
        self.kl_div = nn.KLDivLoss(reduction='none')  # we put no reduction operation, since it will be done manually
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.ignore_index = ignore_index
        self.rank = rank

    # pred (FloatTensor) [batch_size, seq_len, n_classes]
    # target (LongTensor): [batch_size, seq_len]
    # note: if smoothing_coeff is zero, this is equivalent to the cross entropy loss
    def compute_label_smoothing_loss(self, pred, target):
        pred = self.log_softmax(pred)

        batch_size, seq_len, num_classes = pred.shape
        uniform_confidence = self.smoothing_coeff / (num_classes - 1)  # minus one: PAD
        confidence = 1 - self.smoothing_coeff
        one_hot = torch.full((num_classes,), uniform_confidence).to(self.rank)
        model_prob = one_hot.repeat(batch_size, seq_len, 1)
        model_prob.scatter_(2, target.unsqueeze(2), confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(2), 0)

        tot_loss_tensor = self.kl_div(pred, model_prob)  # loss for each [batch_size, max_seq_len, num_classes]

        # divide the loss of each sequence by the number of non pads
        pads_matrix = torch.as_tensor(target == self.ignore_index)
        tot_loss_tensor.masked_fill_(pads_matrix.unsqueeze(2), 0.0)  # zero out all pad rows
        non_pads_vector = (~pads_matrix).sum().type(torch.cuda.FloatTensor)
        tot_loss = tot_loss_tensor.sum() / non_pads_vector

        return tot_loss

    def forward(self, pred, target):
        loss = self.compute_label_smoothing_loss(pred,
                                                 target)
        return loss