import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import device, print_freq, num_workers, grad_clip, root, filelists_train, param_fp_train, filelists_val, \
    param_fp_val
from misc import save_checkpoint, AverageMeter, get_logger, get_learning_rate, clip_gradient
from models import mobilenet_1
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from wpdc_loss import WPDCLoss


def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    # general
    parser.add_argument('--end-epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--base_lr', type=float, default=0.02, help='start learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--milestones', default='30,40', type=str)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    milestones = eval(milestones)

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = mobilenet_1()
        model = nn.DataParallel(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = WPDCLoss()

    # Custom dataloaders
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    train_dataset = DDFADataset(root=root,
                                filelists=filelists_train,
                                param_fp=param_fp_train,
                                transform=transforms.Compose([ToTensorGjz(), normalize]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_dataset = DDFADataset(root=root,
                                filelists=filelists_val,
                                param_fp=param_fp_val,
                                transform=transforms.Compose([ToTensorGjz(), normalize]))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)

    cudnn.benchmark = True

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.milestones)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)

        lr = get_learning_rate(optimizer)
        writer.add_scalar('model/learning_rate', lr, epoch)
        print('\nCurrent effective learning rate: {}\n'.format(lr))

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterion=criterion,
                           logger=logger)

        writer.add_scalar('model/valid_loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (input, target) in enumerate(train_loader):
        # Move to GPU, if available
        input = input.to(device)
        target = target.to(device)

        # Forward prop.
        output = model(input)

        # Calculate loss
        loss = criterion(output, target)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), input.size(0))

        # Print status
        if i % print_freq == 0:
            if i % print_freq == 0:
                status = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})'

                logger.info(status)

    return losses.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()

    with torch.no_grad():
        # Batches
        for i, (input, target) in enumerate(valid_loader):
            # Move to GPU, if available
            input = input.to(device)
            target = target.to(device)

            # Forward prop.
            output = model(input)

            # Calculate loss
            loss = criterion(output, target)

            # Keep track of metrics
            losses.update(loss.item(), input.size(0))

    # Print status
    logger.info(f'Validation\t Loss {losses.avg:.5f}\n')

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
