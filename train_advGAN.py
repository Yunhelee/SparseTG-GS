import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN_ImageNet import AdvGAN_Attack
import sys
from datasets import *

sys.path.append('../../')
import resnet50

csv_file = '../../datasets_def/nrs_train_imagenet_resnet.csv'
img_path = '../../datasets_def/train_imagenet_resnet/'

num_classes = 1000
image_nc = 3
epochs = 100
batch_size = 32
BOX_MIN = 0
BOX_MAX = 1
input_size = 299
# train advGAN

transform = transforms.Compose([
#     transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

model = resnet50.resnet50(pretrained=False)
model.load_state_dict(torch.load('../../pretrain/resnet50-dict(76).pkl'))
model.eval()
model.cuda()

train_data = ImageDataset(csv_file, img_path, transform)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    device = torch.device("cuda" if (True and torch.cuda.is_available()) else "cpu")

    advGAN = AdvGAN_Attack(device, model, num_classes, BOX_MIN, BOX_MAX)
    advGAN.train(train_loader, epochs)
