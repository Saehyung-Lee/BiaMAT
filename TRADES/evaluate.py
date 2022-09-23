from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--BiaMAT', action='store_true')
parser.add_argument('--primary', type=str, default='cifar10',
                    choices=('cifar10'))
parser.add_argument('--auxiliary', type=str, default='imagenet',
                    choices=('imagenet'))
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=100,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random', action='store_true')
parser.add_argument('--data-dir', type=str, default='data',
                    help='data for white-box attack evaluation')
parser.add_argument('--model-path',
                    help='model for white-box attack evaluation')
parser.add_argument('--attack-method',
                    default='all',
                    help='attack method')
parser.add_argument('--gpuid', nargs=2, type=str)

args = parser.parse_args()
print(args.model_path)

if args.auxiliary == 'imagenet':
    num_classes_aug = 1000

if args.BiaMAT:
    from models.wideresnet_BiaMAT import *
else:
    from models.wideresnet import *

if args.multi_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
if args.primary == 'cifar10':
    num_classes= 10
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def one_hot_tensor(y_batch_tensor, num_classes=10):
    if y_batch_tensor.dim() != 1:
        return y_batch_tensor
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)

    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd

def _cw_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)


    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            logits = model(X_pgd)
            label_mask = one_hot_tensor(y, num_classes=num_classes)
            correct_logit = torch.sum(label_mask * logits, dim=1)
            wrong_logit, _ = torch.max((torch.ones_like(label_mask).to(device) - label_mask) * logits - 1e4*label_mask, dim=1)
            loss = -F.relu(correct_logit - wrong_logit + 50.0 * (torch.ones_like(wrong_logit).to(device)))
            loss = loss.mean()
            # loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader, attack):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    print('attack method: %s'%attack)
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        if attack == 'cw':
            err_natural, err_robust = _cw_whitebox(model, X, y)
        elif attack == 'pgd':
            err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total / 10000)
    print('robust_err_total: ', robust_err_total / 10000)

def main():
    # white-box attack
    print('white-box attack')
    model = WideResNet(num_classes=num_classes, num_classes_aug=num_classes_aug).to(device)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)  ## added for multi-GPU
    model.load_state_dict(torch.load(args.model_path))
    if args.attack_method == 'all':
        eval_adv_test_whitebox(model, device, test_loader, 'pgd')
        eval_adv_test_whitebox(model, device, test_loader, 'cw')
    else:
        eval_adv_test_whitebox(model, device, test_loader, args.attack_method)


if __name__ == '__main__':
    main()
