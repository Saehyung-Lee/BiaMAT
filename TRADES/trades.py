import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def one_hot_tensor(y_batch_tensor, num_classes):
    if y_batch_tensor.dim() != 1:
        return y_batch_tensor
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)

    if torch.sum(y_batch_tensor < 0) > 0 or torch.sum(y_batch_tensor >= num_classes) > 0:
        print(y_batch_tensor)
        assert False

    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, target), 1)
        return loss

def trades_loss_aux_out(model,
                x_natural,
                y,
                y_random,
                split,
                optimizer,
                n_class=10,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                alpha=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_ce = softCrossEntropy()
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits_nat = model(x_natural)
    logits = model(x_adv)

    loss_natural_pri = F.cross_entropy(logits_nat[:split[0]], y)
    loss_natural_aux_out = criterion_ce(logits_nat[split[0]:], y_random)

    loss_robust_pri = (1.0 / split[0]) * criterion_kl(F.log_softmax(logits[:split[0]], dim=1), F.softmax(logits_nat[:split[0]], dim=1))
    loss_robust_aux_out = (1.0 / split[1]) * criterion_kl(F.log_softmax(logits[split[0]:], dim=1), F.softmax(logits_nat[split[0]:], dim=1))

    loss_aux = loss_natural_aux_out + beta * loss_robust_aux_out
    loss_pri = loss_natural_pri + beta * loss_robust_pri
    loss = loss_pri + alpha * loss_aux
    return loss

def trades_loss_BiaMAT(model,
                     x_natural,
                     y, #integer
                     y_random, #vector
                     y_aux, #interger
                     optimizer,
                     n_class=10,
                     step_size=0.003,
                     epsilon=0.031,
                     perturb_steps=10,
                     beta=1.0,
                     distance='l_inf',
                     split=None, #[num_in, num_aux_out, num_aux_in]
                     alpha=1.0,
                     ):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_ce = softCrossEntropy(reduce=False)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_nat, logits_aux_nat = model.forward_aux(x_natural, [split[0]+split[1], split[2]])
                logits, logits_aux = model.forward_aux(x_adv, [split[0]+split[1], split[2]])
                loss_kl = criterion_kl(F.log_softmax(logits, dim=1),
                                       F.softmax(logits_nat, dim=1))
                loss_kl += criterion_kl(F.log_softmax(logits_aux, dim=1),
                                       F.softmax(logits_aux_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv = Variable(x_adv)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits_nat, logits_aux_nat = model.forward_aux(x_natural, [split[0]+split[1], split[2]])
    logits, logits_aux = model.forward_aux(x_adv, [split[0]+split[1], split[2]])

    loss_natural_pri = F.cross_entropy(logits_nat[:split[0]], y)
    loss_natural_aux_out = torch.sum(criterion_ce(logits_nat[split[0]:], y_random))

    loss_robust_pri = (1.0 / split[0]) * criterion_kl(F.log_softmax(logits[:split[0]], dim=1), F.softmax(logits_nat[:split[0]], dim=1))
    loss_robust_aux_out = criterion_kl(F.log_softmax(logits[split[0]:], dim=1), F.softmax(logits_nat[split[0]:], dim=1))
    
    loss_natural_aux_in = F.cross_entropy(logits_aux_nat, y_aux, reduction='sum')
    loss_robust_aux_in = criterion_kl(F.log_softmax(logits_aux, dim=1), F.softmax(logits_aux_nat, dim=1))

    loss_aux_out = loss_natural_aux_out + beta * loss_robust_aux_out
    loss_aux_in = loss_natural_aux_in + beta * loss_robust_aux_in
    
    loss_aux = (1./(split[1]+split[2])) * (loss_aux_out + loss_aux_in)
    loss_pri = loss_natural_pri + beta * loss_robust_pri
    loss = loss_pri + alpha * loss_aux

    return loss, alpha * loss_aux, loss_aux_out, loss_aux_in

def trades_loss_BiaMAT_naive(model,
                     x_natural,
                     y,
                     y_aux,
                     optimizer,
                     n_class=10,
                     step_size=0.003,
                     epsilon=0.031,
                     perturb_steps=10,
                     beta=1.0,
                     distance='l_inf',
                     split=None,
                     alpha=1.0,
                     ):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_nat, logits_aux_nat = model.forward_aux(x_natural, split)  # size split[0], split[1]
                logits, logits_aux = model.forward_aux(x_adv, split)  # size split[0], split[1]
                loss_kl = criterion_kl(F.log_softmax(logits, dim=1),
                                       F.softmax(logits_nat, dim=1))  # size split[0]
                loss_kl += criterion_kl(F.log_softmax(logits_aux, dim=1),
                                       F.softmax(logits_aux_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv = Variable(x_adv)

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits_nat, logits_aux_nat = model.forward_aux(x_natural, split)
    logits, logits_aux = model.forward_aux(x_adv, split)

    loss_natural_pri = F.cross_entropy(logits_nat, y)
    loss_natural_aux = F.cross_entropy(logits_aux_nat, y_aux)
    loss_robust_pri = (1.0 / split[0]) * criterion_kl(F.log_softmax(logits, dim=1),
                                                  F.softmax(logits_nat, dim=1))
    loss_robust_aux = (1.0 / split[1]) * criterion_kl(F.log_softmax(logits_aux, dim=1),
                                                  F.softmax(logits_aux_nat, dim=1))
    loss_default = loss_natural_pri + beta * loss_robust_pri
    loss_aux = loss_natural_aux + beta * loss_robust_aux
    loss = loss_default + alpha * loss_aux

    return loss, alpha * loss_aux

def trades_loss_BiaMAT_ce(model,
                     x_natural,
                     y, #integer
                     y_random, #vector
                     y_aux, #interger
                     optimizer,
                     n_class=10,
                     step_size=0.003,
                     epsilon=0.031,
                     perturb_steps=10,
                     beta=1.0,
                     distance='l_inf',
                     split=None, #[num_in, num_aux_out, num_aux_in]
                     alpha=1.0,
                     ):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_ce = softCrossEntropy(reduce=False)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_nat, logits_aux_nat = model.forward_aux(x_natural, [split[0]+split[1], split[2]])
                logits, logits_aux = model.forward_aux(x_adv, [split[0]+split[1], split[2]])
                loss_kl = criterion_kl(F.log_softmax(logits, dim=1),
                                       F.softmax(logits_nat, dim=1))
                loss_kl += F.cross_entropy(logits_aux, y_aux, reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv = Variable(x_adv)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits_nat, logits_aux_nat = model.forward_aux(x_natural, [split[0]+split[1], split[2]])
    logits, logits_aux = model.forward_aux(x_adv, [split[0]+split[1], split[2]])

    loss_natural_pri = F.cross_entropy(logits_nat[:split[0]], y)
    loss_natural_aux_out = torch.sum(criterion_ce(logits_nat[split[0]:], y_random))

    loss_robust_pri = (1.0 / split[0]) * criterion_kl(F.log_softmax(logits[:split[0]], dim=1), F.softmax(logits_nat[:split[0]], dim=1))
    loss_robust_aux_out = criterion_kl(F.log_softmax(logits[split[0]:], dim=1), F.softmax(logits_nat[split[0]:], dim=1))
    
    loss_aux_out = loss_natural_aux_out + beta * loss_robust_aux_out
    loss_aux_in = F.cross_entropy(logits_aux, y_aux, reduction='sum')
    
    loss_aux = (1./(split[1]+split[2])) * (loss_aux_out + loss_aux_in)
    loss_pri = loss_natural_pri + beta * loss_robust_pri
    loss = loss_pri + alpha * loss_aux

    return loss, alpha * loss_aux, loss_aux_out, loss_aux_in

def trades_loss_BiaMAT_naive_ce(model,
                     x_natural,
                     y,
                     y_aux,
                     optimizer,
                     n_class=10,
                     step_size=0.003,
                     epsilon=0.031,
                     perturb_steps=10,
                     beta=1.0,
                     distance='l_inf',
                     split=None,
                     alpha=1.0,
                     ):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_nat, logits_aux_nat = model.forward_aux(x_natural, split)  # size split[0], split[1]
                logits, logits_aux = model.forward_aux(x_adv, split)  # size split[0], split[1]
                loss_kl = criterion_kl(F.log_softmax(logits, dim=1),
                                       F.softmax(logits_nat, dim=1))  # size split[0]
                loss_kl += F.cross_entropy(logits_aux, y_aux)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv = Variable(x_adv)

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits_nat, logits_aux_nat = model.forward_aux(x_natural, split)
    logits, logits_aux = model.forward_aux(x_adv, split)

    loss_natural_pri = F.cross_entropy(logits_nat, y)
    loss_robust_pri = (1.0 / split[0]) * criterion_kl(F.log_softmax(logits, dim=1),
                                                  F.softmax(logits_nat, dim=1))
    loss_default = loss_natural_pri + beta * loss_robust_pri
    loss_aux = F.cross_entropy(logits_aux, y_aux)
    loss = loss_default + alpha * loss_aux

    return loss, alpha * loss_aux
