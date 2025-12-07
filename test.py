import numpy
import numpy as np
import torch
import models
import sys
from datasets import *

sys.path.append("../../")
from VGG16 import *

csv_file = '../../datasets_def/nrs_test_cifar_vgg.csv'
img_path = '../../datasets_def/test_cifar_vgg/'

batch_size = 1

transform = transforms.Compose([
    # transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# netT = inception_v3.inception_v3(pretrained=False)
# netT.load_state_dict(torch.load('../../pretrain/inception_v3_google-1a9a5a14.pth'))

train_data = ImageDataset(csv_file, img_path, transform)
loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

pretrained_generator_path = './models/netG_epoch_100(vgg).pth'
pretrained_G = models.Generator().cuda()
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

success = 0
for idx, (images, label) in enumerate(loader):

    images = images.cuda()
    label = label.cuda()
    advImgs = images[:, :, 0:32, :]
    imgs = images[:, :, 32:, :]
    mask = pretrained_G(advImgs)
    advNoise = imgs - advImgs
    advImgs = imgs - mask * advNoise
    output = model(advImgs)
    _, pred = torch.max(output, 1)
    if pred == label:
        success += 1
#     print((mask < 0.5).sum())
#     print(torch.norm(mask,0).item())
print(success)