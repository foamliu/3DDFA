import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import scipy.io as sio
from config import device
from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.estimate_pose import parse_pose
from utils.inference import predict_68pts, parse_roi_box_from_bbox, predict_dense, dump_to_ply, get_suffix
from misc import ensure_folder


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

    print('param: ' + str(param))

    # 68 pts
    bbox = [0, 0, 120, 120]
    roi_box = parse_roi_box_from_bbox(bbox)
    pts68 = predict_68pts(param, roi_box)
    print('pts68: ' + str(pts68))

    P, pose = parse_pose(param)
    print('P: ' + str(P))
    print('pose: ' + str(pose))

    vertices = predict_dense(param, roi_box)
    print('vertices: ' + str(vertices))
    print('vertices.shape: ' + str(vertices.shape))

    ensure_folder('result')
    suffix = get_suffix(filename)
    print('suffix: ' + suffix)
    tri = sio.loadmat('visualize/tri.mat')['tri']
    dump_to_ply(vertices, tri, 'result/{}.ply'.format(suffix))
