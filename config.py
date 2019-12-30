import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 224
num_classes = 62

root = 'data/train_aug_120x120'
filelists_train = 'train.configs/train_aug_120x120.list.train'
filelists_val = 'train.configs/train_aug_120x120.list.val'
param_fp_train = 'train.configs/param_all_norm.pkl'
param_fp_val = 'train.configs/param_all_norm_val.pkl'

# Training parameters
num_workers = 6  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 50  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
