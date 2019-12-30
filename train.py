import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from config import device, print_freq, num_workers, grad_clip, root, filelists_train, param_fp_train, filelists_val, param_fp_val
from misc import parse_args, save_checkpoint, AverageMeter, get_logger, get_learning_rate, clip_gradient
from models import mobilenet_1
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from vdc_loss import VDCLoss


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

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

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
    criterion = VDCLoss()

    # Custom dataloaders
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    train_dataset = DDFADataset(root=root,
                                filelists=filelists_train,
                                param_fp=param_fp_train,
                                transform=transforms.Compose([ToTensorGjz(), normalize]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers)
    valid_dataset = DDFADataset(root=root,
                                filelists=filelists_val,
                                param_fp=param_fp_val,
                                transform=transforms.Compose([ToTensorGjz(), normalize]))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=num_workers)

    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
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
        scheduler.step(epoch)


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
