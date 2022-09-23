from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from models.wideresnet_BiaMAT import *
from dataset import *
from trades import *
import time
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--primary', type=str, default='cifar10',
                    choices=('cifar10', 'cifar100'))
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=110, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-schedule', type=str, default='bag_of_tricks',
                    choices=('trades', 'bag_of_tricks'),
                    help='Learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='a hyperparameter for BiaMAT')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='a hyperparameter for BiaMAT')
parser.add_argument('--warmup', type=int, default=5,
                    help='a hyperparameter for BiaMAT')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', type=str,
                    help='directory of model for saving checkpoint')
parser.add_argument('--data-dir', type=str, default='data',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--aux-dataset-dir', default='Imagenet32_train')
parser.add_argument('--load-model-dir',
                   help='directory of model for saving checkpoint')
parser.add_argument('--load-epoch', type=int, default=0, metavar='N',
                    help='load epoch')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
print(args)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

'''''''''
Loading primary dataset
'''''''''
if args.primary == 'cifar10':
    num_classes=10
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
elif args.primary == 'cifar100':
    num_classes=100
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
'''''''''
Loading auxiliary dataset
'''''''''
aux_dataset = Load_dataset(args.aux_dataset_dir)
print('dataset size : ', aux_dataset.image.shape, aux_dataset.label.shape)
print('dataset min, max : ', np.min(aux_dataset.image), np.max(aux_dataset.image)) # 0 255
print('label min, max : ', np.min(aux_dataset.label), np.max(aux_dataset.label)) # 0 999
print("alpha : %f, gamma: %f"%(args.alpha, args.gamma))
cur_trainset = Auxiliary(aux_dataset.image, aux_dataset.label, transform_train) # 0, 125
train_loader_aug = torch.utils.data.DataLoader(cur_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta
                          )
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def biamat_gate(data, data_aux, target, target_aux, model, warmup, threshold):
    data_concat = torch.cat((data, data_aux), dim=0)
    if warmup:
        return data_concat, target, None, target_aux, [len(target), 0, len(target_aux)], threshold , 0., 0.
    model.eval()
    if threshold == 0:
        logits = model(data_concat)
        confidences, _ = torch.max(F.softmax(logits.detach(), dim=1).detach(), dim=1)
        mean_confi_pri = torch.mean(confidences[:len(target)].detach())
        confi_aux = confidences[len(target):].detach()
        mean_confi_aux = torch.mean(confi_aux)
        threshold = mean_confi_pri*args.gamma
        mask = confi_aux < threshold
    else:
        logits = model(data_aux)
        confidences, _ = torch.max(F.softmax(logits.detach(), dim=1).detach(), dim=1)
        mean_confi_aux = torch.mean(confidences)
        mask = confidences < threshold
    num_out = torch.sum(mask.long())
    if num_out == len(mask):
        target_random = torch.ones(len(target_aux), num_classes)*(1./num_classes)
        return data_concat, target, target_random.cuda(), None, [len(target), len(target_random), 0], threshold, mean_confi_aux
    elif num_out == 0:
        return data_concat, target, None, target_aux, [len(target), 0, len(target_aux)], threshold, mean_confi_aux
    else:
        data_aux_out = data_aux[mask].detach()
        data_aux_in = data_aux[~mask].detach()
        data_aux_reorder = torch.cat([data_aux_out, data_aux_in], dim=0)
        data_concat[-len(target_aux):] = data_aux_reorder.detach()
        target_random = torch.ones(num_out, num_classes)*(1./num_classes)
        target_aux = target_aux[~mask]
        return data_concat, target, target_random.cuda(), target_aux, [len(target), len(target_random), len(target_aux)], threshold, mean_confi_aux, args.alpha

def train_BiaMAT(args, model, device, train_loader, train_loader_aug, aug_iterator, optimizer, epoch, warmup, threshold):
    model.train()
    biamat=0
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            data_aug, target_aug = next(aug_iterator)
        except StopIteration:
            aug_iterator = iter(train_loader_aug)
            data_aug, target_aug = next(aug_iterator)
        data, target, target_random, target_aux, split, threshold, mean_confi_aux, alpha = biamat_gate(data.to(device), data_aug.to(device), target.to(device), target_aug.to(device), model, warmup, threshold)
        optimizer.zero_grad()
        # calculate robust loss
        if split[1] == 0:
            if args.primary == 'cifar10':
                loss, loss_aux = trades_loss_BiaMAT_naive(model=model,
                                   x_natural=data,
                                   y=target,
                                   y_aux=target_aux,
                                   split=[split[0], split[2]],
                                   optimizer=optimizer,
                                   step_size=args.step_size,
                                   epsilon=args.epsilon,
                                   perturb_steps=args.num_steps,
                                   beta=args.beta,
                                   alpha=alpha
                                  )
            elif args.primary == 'cifar100':
                loss, loss_aux = trades_loss_BiaMAT_naive_ce(model=model,
                                   x_natural=data,
                                   y=target,
                                   y_aux=target_aux,
                                   split=[split[0], split[2]],
                                   optimizer=optimizer,
                                   step_size=args.step_size,
                                   epsilon=args.epsilon,
                                   perturb_steps=args.num_steps,
                                   beta=args.beta,
                                   alpha=alpha
                                  )
            loss_aux_out = torch.tensor(0)
            loss_aux_in = torch.tensor(0)
        elif split[2] == 0:
            loss = trades_loss_aux_out(model=model,
                               x_natural=data,
                               y=target,
                               y_random=target_random,
                               split=split[:2],
                               optimizer=optimizer,
                               step_size=args.step_size,
                               epsilon=args.epsilon,
                               perturb_steps=args.num_steps,
                               beta=args.beta,
                               alpha=alpha
                              )
            loss_aux = 0
        else:
            if args.primary == 'cifar10':
                loss, loss_aux, loss_aux_out, loss_aux_in = trades_loss_BiaMAT(model=model,
                                   x_natural=data,
                                   y=target,
                                   y_random=target_random,
                                   y_aux=target_aux,
                                   split=split,
                                   optimizer=optimizer,
                                   step_size=args.step_size,
                                   epsilon=args.epsilon,
                                   perturb_steps=args.num_steps,
                                   beta=args.beta,
                                   alpha=alpha
                                  )
            elif args.primary == 'cifar100':
                loss, loss_aux, loss_aux_out, loss_aux_in = trades_loss_BiaMAT_ce(model=model,
                                   x_natural=data,
                                   y=target,
                                   y_random=target_random,
                                   y_aux=target_aux,
                                   split=split,
                                   optimizer=optimizer,
                                   step_size=args.step_size,
                                   epsilon=args.epsilon,
                                   perturb_steps=args.num_steps,
                                   beta=args.beta,
                                   alpha=alpha
                                  )
        loss.backward()
        biamat += split[2] / sum(split[1:])
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss Auxiliary: {:.6f}, loss_aux_out: {:.6f} ,loss_aux_in: {:.6f}\tbiamat ratio: {:.6f}, mean_confi_aux: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_aux.item(), loss_aux_out.item(), loss_aux_in.item(), biamat / args.log_interval, mean_confi_aux))
            biamat = 0
    return aug_iterator, threshold

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if args.lr_schedule == 'trades':
        if epoch >= int(args.epochs * 0.75):
            lr = args.lr * 0.1
        if epoch >= int(args.epochs) * 0.9:
            lr = args.lr * 0.01
    elif args.lr_schedule == 'bag_of_tricks':
        if epoch >= int(args.epochs * (100./110.)):
            lr = args.lr * 0.1
        if epoch >= int(args.epochs * (105./110.)):
            lr = args.lr * 0.01
    else:
        print('specify lr scheduler!!')
        sys.exit(1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model
    model = WideResNet(num_classes=num_classes, num_classes_aug=aux_dataset.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.load_epoch != 0:
        print('resuming ... ', args.load_model_dir)
        f_path = os.path.join(args.load_model_dir)
        checkpoint = torch.load(f_path)
        model.load_state_dict(checkpoint)
        eval_test(model, device, test_loader)

    init_time = time.time()

    aug_iterator = iter(train_loader_aug)
    warmup = True
    threshold = 0
    for epoch in range(args.load_epoch + 1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        elapsed_time = time.time() - init_time
        print('elapsed time : %d h %d m %d s' % (elapsed_time / 3600, (elapsed_time % 3600) / 60, (elapsed_time % 60)))
        if epoch >= args.warmup+1:
            warmup = False
        aug_iterator, threshold = train_BiaMAT(args, model, device, train_loader, train_loader_aug, aug_iterator, optimizer, epoch, warmup, threshold)

        # save checkpoint
        if epoch <= 10:
            eval_test(model, device, test_loader)

        if (epoch % args.save_freq == 0) and (epoch >= int(args.epochs * 0.9)):
            # evaluation on natural examples
            print('================================================================')
            eval_test(model, device, test_loader)
            print('================================================================')
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
        torch.save(model.state_dict(),
                   os.path.join(model_dir, 'model-wideres-latest.pt'.format(epoch)))



if __name__ == '__main__':
    main()
