import torch.nn as nn
import torch
import numpy as np
import models_ImageNet as models
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import os
from datasets import *
import sys

models_path = './models/'
criterion = nn.CrossEntropyLoss()
N = 299 * 299 * 3
kappa = 100
batch_size = 32

real_label = torch.ones(batch_size).long().cuda()  ## 定义真实的图片label为1
fake_label = torch.zeros(batch_size).long().cuda()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def CWLoss(logits, target, kappa=0, tar=True):
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].cuda())

    real = torch.sum(target_one_hot * logits, 1)
    other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if tar:
        return torch.sum(torch.max(other - real, kappa))
    else:
        return torch.sum(torch.max(real - other, kappa))


class AdvGAN_Attack:
    def __init__(self, device, model, model_num_labels, box_min, box_max):
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.box_min = box_min
        self.box_max = box_max

        self.netG = models.Generator().to(device)
        self.netDisc = models.Discriminator().to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, imgs, labels):
        # optimize D
        for i in range(1):
            mask = self.netG(x)
            advNoise = imgs - x

            # add a clipping trick
            # adv_images = torch.clamp(mask, -0.05, 0.05) + x
            adv_images = imgs - mask * advNoise
            #             adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)

            #             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real = criterion(pred_real, torch.ones(pred_real.size(0)).long().cuda())
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            #             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake = criterion(pred_fake, torch.zeros(pred_fake.size(0)).long().cuda())
            loss_D_fake.backward()

            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            # loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake = criterion(pred_fake, torch.ones(pred_real.size(0)).long().cuda())
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            # L_hinge
            C = 0.1
            loss_perturb = torch.mean(torch.norm(mask.view(mask.shape[0], -1), 1, dim=1))
            #             loss_perturb = torch.sum(mask)
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images)
            # probs_model = F.softmax(logits_model, dim=1)
            # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # # C&W loss function
            # real = torch.sum(onehot_labels * probs_model, dim=1)
            # other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            # zeros = torch.zeros_like(other)
            # loss_adv = torch.max(real - other, zeros)
            # loss_adv = torch.sum(loss_adv)
            loss_adv = CWLoss(logits_model, labels, -kappa, False)

            # Lambda = 1
            # loss_adv += Lambda * torch.norm(mask, 1) / N

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10000
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_loader, epochs):
        for epoch in range(1, epochs + 1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
                scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=10, gamma=0.2)
                scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=10, gamma=0.2)
            if epoch == 100:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_loader, start=0):
                images, labels = data
                advImgs = images[:,:,0:299,:]
                imgs = images[:,:,299:,:]

                imgs = imgs.cuda()
                advImgs = advImgs.cuda()
                labels = labels.cuda()
                imgs = Variable(imgs, requires_grad=True)
                # advImgs = attack(imgs, labels)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(advImgs, imgs, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_loader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                   loss_perturb_sum / num_batch, loss_adv_sum / num_batch))
            #             scheduler_G.step()
            #             scheduler_D.step()

            # save generator
            if epoch % 20 == 0:
                netG_file_name = models_path + 'netG_epoch_ImageNet_ResNet_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
