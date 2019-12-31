import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from config import device
from utils.ddfa import ToTensorGjz, NormalizeGjz

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module

    cudnn.benchmark = True
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    filename = 'images/0000.png'
    img = cv.imread(filename)
    img = cv.resize(img, (120, 120), interpolation=cv.INTER_LINEAR)
    input = transform(img).unsqueeze(0)
    input = input.to(device)

    with torch.no_grad():
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

    print(param)
