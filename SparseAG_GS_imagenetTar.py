import torch
from skimage.segmentation import slic

from options import BaseOptions
import numpy as np
from datasets import *
import torchvision
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
from torchvision.utils import save_image
from PIL import Image
import time
import models_ImageNet as models
from torchvision.transforms import ToPILImage
import random

sys.path.append('../../')
import resnet50

cudnn.benchmark = True

random.seed(0)
L = random.sample(range(0, 1000), 50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps = 0.050
kappa = 100
n_segments = [600]
im_size = 299

csv_file = '../../nrs_val.csv'
img_path = '../../val/'

transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
])

pretrained_generator_path = './models/netG_epoch_ImageNet_ResNet_100.pth'
pretrained_G = models.Generator().cuda()
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()  # input/output表示获取指定层的输入/输出

    return hook


def main():
    netT = resnet50.resnet50(pretrained=False)
    netT.load_state_dict(torch.load('../../pretrain/resnet50-dict(76).pkl'))
    netT.eval()
    netT.cuda()

    test_data = ImageDataset(csv_file, img_path, transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    total_time = 0
    total_group = [0]
    total_best_group = [0]
    total_wrost_group = [0]

    total_reduce = 0
    length = 0
    success = 0
    total_norm_inf = 0
    total_norm_2 = 0
    total_norm_1 = 0
    total_norm_0 = 0
    total_best_case_0 = 0
    total_best_case_1 = 0
    total_best_case_2 = 0
    total_best_case_inf = 0
    total_wrong_case_0 = 0
    total_wrong_case_1 = 0
    total_wrong_case_2 = 0
    total_wrong_case_inf = 0
    total_running_time = 0
    pretrained_G.decoder.register_forward_hook(get_activation('decoder'))

    boost = False
    cnt = 0
    print('The boost is : ', boost)
    for idx, (images, labels) in enumerate(test_loader):
        if idx not in L:
            continue

        print('\nThe current image index is : ', idx + 1)

        real_A = Variable(images, requires_grad=True)
        real_A = real_A.cuda()
        image_names = labels.cuda()

        logist_B = netT(real_A)
        _, pre = torch.max(logist_B, dim=1)
        if pre.cpu().data.float() != image_names.cpu().data.float():
            print('Prediction Wrong !')
            continue
        length += 1

        logist_B = netT(real_A)
        _, target = torch.max(logist_B, dim=1)
        # Increasing

        pert_min = 100000
        pert_max = 0
        best_adv_sample = 0
        wrong_adv = 0
        segments = superpixel(real_A, n_segments[0])  # 1x3x32x32

        while True:
            select_tar = random.sample(range(0, 1000), 9)
            if image_names not in select_tar:
                break
        count = 0
        for tar in select_tar:
            if tar == image_names:
                continue
            tar = 505
            if count == 1:
                break
            count += 1
            mask = torch.zeros(1, 3, im_size, im_size).cuda()
            adv_mask = torch.zeros_like(mask)

            start_time = time.time()
            # Increasing
            segments = segments.cuda()
            max_iter = torch.max(segments)

            adv = real_A.clone()
            target = torch.IntTensor([tar]).cuda()
            print('The target is : ', target.item())
            print('The maxiter is: ', max_iter)

            adv, adv_mask, mask, boolean = increase(real_A, adv, segments, netT, target, mask, boost, adv_mask=adv_mask)
            save_image(real_A, adv)
            if not boolean:
                print('Attack failed!')
                continue
            adv_noise = real_A - adv
            norm_0 = torch.norm(adv_noise, p=0)
            logits = netT(adv.to(device))
            pred = torch.argmax(logits, dim=1)
            print('The increase results is : ', norm_0.item(), " and the prediction label is : ", pred.item())

            best_group = cal_group(real_A, adv, segments.cpu())
            const = 0
            const_temp = 0
            iter_num = 0
            best_adv = adv
            reduce_prop = 0.1
            max_prop = 0.2
            while const <= 10 and iter_num <= 20:
                print('The current number is : ', iter_num)

                former = cal_group(real_A, adv, segments.cpu())

                advimportance = pretrained_G(adv)
                adv = reduce(real_A, adv, netT, target, adv_mask, advimportance, mask, True)

                group_num = cal_group(real_A, adv, segments.cpu())
                if group_num < best_group:
                    best_group = group_num
                    best_adv = adv

                adv, adv_mask, mask, boolean = increase(real_A, adv, segments, netT, target, mask, boost,
                                                        adv_mask=adv_mask,
                                                        fine_tuning=True)

                latter = cal_group(real_A, adv, segments.cpu())
                print('The L0 norm is : ', torch.norm(real_A - adv, 0).item())
                print('The former and the latter : ', former, latter)
                if latter >= former:
                    const += 1
                    const_temp += 1
                else:
                    const = 0
                if latter <= best_group:
                    best_group = latter
                    best_adv = adv
                if const_temp >= 7:
                    const_temp = 0
                    reduce_prop *= 1.5
                    reduce_prop = np.clip(reduce_prop, 0, max_prop)
                iter_num += 1
            adv = best_adv

            print('The final norm 0 is : ', torch.norm(real_A - adv, 0).item())
            advimportance = pretrained_G(adv)
            adv = reduce(real_A, adv, netT, target, adv_mask, advimportance, mask)
            logits = netT(adv.to(device))
            pred1 = torch.argmax(logits, dim=1)
            print(torch.norm(real_A - adv, 2).item())

            alpha = 1
            a = [alpha + i * 3 / 10 for i in range(0, 30)]
            for item in a:
                advimportance = torch.nn.functional.sigmoid(item * activation['decoder'])

                adv_noise = real_A - adv
                adv_noise = adv_noise * advimportance
                temp_adv = real_A - adv_noise
                logits = netT(temp_adv.to(device))
                pred2 = torch.argmax(logits, dim=1)
                if pred1 == pred2:
                    adv = temp_adv
                    break

            end_time = time.time()

            logits = netT(adv.to(device))
            pred = torch.argmax(logits, dim=1)
            print('prediction label: ', pred.item(), ' True label: ', target.item())
            if pred == target:
                success += 1
            else:
                print('failed!')

            sum = 0
            adv_noise = real_A - adv
            pert_norm_inf = torch.norm(adv_noise, float('inf'))
            pert_norm_2 = torch.norm(adv_noise)
            pert_norm_1 = torch.norm(adv_noise, p=1)
            pert_norm_0 = torch.norm(adv_noise, p=0)
            total_norm_inf += pert_norm_inf
            total_norm_2 += pert_norm_2
            total_norm_1 += pert_norm_1
            total_norm_0 += pert_norm_0
            total_time += (end_time - start_time)
            print('The inf norm perturbation is : ', pert_norm_inf.item())
            print('The 2 norm perturbation is : ', pert_norm_2.item())
            print('The 1 norm perturbation is : ', pert_norm_1.item())
            print('The 0 norm perturbation is " ', pert_norm_0.item())
            print('Time: " ', (end_time - start_time))
            print('Reduce result is : ', torch.norm(adv_noise, p=0).item())
            total_reduce += sum

            if pert_norm_0 < pert_min:
                pert_min = pert_norm_0
                best_adv_sample = adv
            if pert_norm_0 > pert_max:
                pert_max = pert_norm_0
                wrong_adv = adv

            adv_noise = real_A - adv
            adv_noise = adv_noise.cpu()
            cur_group = [0]
            for index, num in enumerate(n_segments):
                segments = superpixel(real_A, num)
                max_iter = torch.max(segments)
                for j in range(1, max_iter + 1):
                    segmentstemp = cal_ss(segments, j)
                    for k in range(3):
                        sum = torch.norm(segmentstemp[0, k, :, :] * adv_noise[0, k, :, :], p=0)
                        if sum != 0:
                            cur_group[index] += 1
            total_group = np.add(total_group, cur_group)
            print('The group nums are:', cur_group)
            sys.stdout.flush()

        adv_best = real_A - best_adv_sample
        adv_wrong = real_A - wrong_adv
        total_best_case_0 += torch.norm(adv_best, p=0)
        total_best_case_1 += torch.norm(adv_best, p=1)
        total_best_case_2 += torch.norm(adv_best)
        total_best_case_inf += torch.norm(adv_best, p=float('inf'))
        total_wrong_case_0 += torch.norm(adv_wrong, p=0)
        total_wrong_case_1 += torch.norm(adv_wrong, p=1)
        total_wrong_case_2 += torch.norm(adv_wrong)
        total_wrong_case_inf += torch.norm(adv_wrong, p=float('inf'))
        print(pert_min, pert_max)
        print('The best L0 norm is ： ', torch.norm(adv_best, p=0))
        print('The wrong L0 norm is ： ', torch.norm(adv_wrong, p=0))

        real_A = real_A.cuda()
        adv_best_noise = real_A - best_adv_sample
        adv_wrost_noise = real_A - wrong_adv
        adv_best_noise = adv_best_noise.cpu()
        adv_wrost_noise = adv_wrost_noise.cpu()
        cur_best_group = [0]
        cur_wrost_group = [0]
        for index, num in enumerate(n_segments):
            segments = superpixel(real_A, num)
            max_iter = torch.max(segments)
            for j in range(1, max_iter + 1):
                segmentstemp = cal_ss(segments, j)
                for k in range(3):
                    sum_best = torch.norm(segmentstemp[0, k, :, :] * adv_best_noise[0, k, :, :], p=0)
                    sum_wrost = torch.norm(segmentstemp[0, k, :, :] * adv_wrost_noise[0, k, :, :], p=0)
                    if sum_best != 0:
                        cur_best_group[index] += 1
                    if sum_wrost != 0:
                        cur_wrost_group[index] += 1
        total_best_group = np.add(total_best_group, cur_best_group)
        total_wrost_group = np.add(total_wrost_group, cur_wrost_group)
        print('The best and wrost group nums are:' + str(cur_best_group) + ' ' + str(cur_wrost_group))

    mean_norm_inf = float(total_norm_inf) / success
    mean_norm_2 = float(total_norm_2) / success
    mean_norm_1 = float(total_norm_1) / success
    mean_nrom_0 = float(total_norm_0) / success
    mean_best_case_0 = float(total_best_case_0) / length
    mean_best_case_1 = float(total_best_case_1) / length
    mean_best_case_2 = float(total_best_case_2) / length
    mean_best_case_inf = float(total_best_case_inf) / length
    mean_wrong_case_0 = float(total_wrong_case_0) / length
    mean_wrong_case_1 = float(total_wrong_case_1) / length
    mean_wrong_case_2 = float(total_wrong_case_2) / length
    mean_wrong_case_inf = float(total_wrong_case_inf) / length
    mean_running_time = float(total_running_time) / success
    mean_group = [item / success for item in total_group]
    mean_best_group = [item / success for item in total_best_group]
    mean_wrong_group = [item / success for item in total_wrost_group]
    filename = 'result.txt'
    with open(filename, 'a') as file_object:
        file_object.write('------------Target Attack ImageNet ResNet------------')
        file_object.write('\n The final inf norm are : ' + str(mean_norm_inf) + '\n')
        file_object.write('The final 2 norm are : ' + str(mean_norm_2) + '\n')
        file_object.write('The final 1 norm are : ' + str(mean_norm_1) + '\n')
        file_object.write('The final 0 norm are : ' + str(mean_nrom_0) + '\n')
        file_object.write('The best inf norm are : ' + str(mean_best_case_inf) + '\n')
        file_object.write('The best 2 norm are : ' + str(mean_best_case_2) + '\n')
        file_object.write('The best 1 norm are : ' + str(mean_best_case_1) + '\n')
        file_object.write('The best 0 norm are : ' + str(mean_best_case_0) + '\n')
        file_object.write('The wrong inf norm are : ' + str(mean_wrong_case_inf) + '\n')
        file_object.write('The wrong 2 norm are : ' + str(mean_wrong_case_2) + '\n')
        file_object.write('The wrong 1 norm are : ' + str(mean_wrong_case_1) + '\n')
        file_object.write('The wrong 0 norm are : ' + str(mean_wrong_case_0) + '\n')
        file_object.write('The length and success is : ' + str(length) + ' ' + str(success) + '\n')
        file_object.write('The mean running time is : ' + str(mean_running_time) + '\n')
        file_object.write('The mean groups are : ' + str(mean_group) + '\n')
        file_object.write('The mean best groups are : ' + str(mean_best_group) + '\n')
        file_object.write('The mean wrong groups are : ' + str(mean_wrong_group) + '\n')

    print('The average inf norm are : ', mean_norm_inf)
    print('The average 2 norm are : ', mean_norm_2)
    print('The average 1 norm are : ', mean_norm_1)
    print('The average 0 norm are : ', mean_nrom_0)
    print("The length and success are: " + str(length) + ' ' + str(success))
    sys.stdout.flush()


def select(param, mask):
    if torch.sum(param * mask) == 0:
        return True
    else:
        return False


def increase(real_A, adv, segments, netT, target, mask, boost, adv_mask=None, fine_tuning=False):
    Iter = 100000
    temp_eps = eps / 2
    max_iter = torch.max(segments)

    loss_adv = CWLoss

    if fine_tuning:
        mask, adv_mask = perturbation_importance(real_A, adv, netT, target, adv_mask, mask, segments.cpu())
        adv_noise = real_A - adv
        adv_noise = adv_noise * mask
        adv = real_A - adv_noise

    for iters in range(Iter):  # iter=10000
        if iters == max_iter * 3:
            return adv, adv_mask, mask, False
        temp_A = Variable(adv.clone(), requires_grad=True)
        logist_B = netT(temp_A)
        _, pre = torch.max(logist_B, dim=1)
        if target.cpu().data.float() == pre.cpu().data.float():
            break
        Loss = loss_adv(logist_B, target, -kappa, False, True) / real_A.size(0)
        netT.zero_grad()
        if temp_A.grad is not None:
            temp_A.grad.data.fill_(0)
        Loss.backward()

        grad = temp_A.grad

        selnum = 25
        with torch.no_grad():
            mask_alt = mask.clone()
            mask_alt = mask_alt.repeat(max_iter * 3 - iters, 1, 1, 1)
            grad_order = dict()
            num = 0
            for i in range(1, max_iter + 1):
                temp_segments = cal_ss(segments=segments, index=i)
                for j in range(3):
                    boolean = select(temp_segments[0, j, :, :], mask[0, j, :, :])
                    if boolean:
                        mask_alt[num, j, :, :] = mask_alt[num, j, :, :] + temp_segments[0, j, :, :]
                        num += 1
            for i in range(0, max_iter * 3 - iters):
                temp_grad = (mask_alt[i] - mask) * grad
                grad_order[i] = torch.mean(temp_grad)
            # grad_order = sorted(grad_order.items(), key=lambda x: x[1], reverse=True)
            grad_order = sorted(grad_order.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            temp_mask = grad_order[0:selnum]
            mask_temp = torch.zeros((selnum, 3, im_size, im_size)).cuda()
            for i in range(selnum):
                mask_temp[i] = mask_alt[temp_mask[i][0]]
            grad = mask_temp * grad

            abs_grad = torch.abs(grad)
            normalized_grad = abs_grad * grad.sign()
            scaled_grad = normalized_grad.mul(temp_eps)
            temp_A_alt = temp_A - scaled_grad

            temp_A_alt = clip(temp_A_alt, real_A, eps)
            adv = torch.clamp(temp_A_alt, 0, 1)

            logist_B = netT(adv.float())
            Loss = loss_adv(logist_B, target, -100, True, True)
            min_loss, min_num = torch.min(Loss, dim=0)
            choose_mask = mask_temp[min_num] - mask
            adv_mask = torch.cat((adv_mask, choose_mask))
            mask = torch.unsqueeze(mask_temp[min_num], 0)

            adv = adv[min_num].float()
            adv = torch.unsqueeze(adv, dim=0)
    if fine_tuning:
        return adv, adv_mask, mask, True
    else:
        return adv, adv_mask[1:], mask, True


def reduce(real_A, adv, netT, target, adv_mask, advimportance, mask, group=False):
    adv_noise = real_A - adv
    abs_noise = torch.abs(adv_noise)

    if group:
        length = adv_mask.size()[0]
        advimportance = advimportance * adv_mask
        importance_dict = dict()
        importance_mean = advimportance.mean(dim=1).mean(dim=1).mean(dim=1)
        for i in range(length):
            importance_dict[i] = importance_mean[i].item()
        importance_dict = sorted(importance_dict.items(), key=lambda x: x[1], reverse=False)

    if group:
        modi_num = adv_mask.size()[0]
    else:
        modi_num = torch.norm(abs_noise, p=0)

    if modi_num > 0:
        if group:
            interval_num = max(modi_num * 0.1, 30)
        else:
            interval_num = max(modi_num * 0.1, 300)
        reduce_idx = 0
        reduce_count = 0
        count = 0
        order = []
        while reduce_idx < modi_num and reduce_count < 3000:
            #             if not group:
            #                 print(reduce_idx, modi_num)
            reduce_count += 1

            adv_noise = real_A - adv

            if group:
                temp_mask = mask - adv_mask[importance_dict[reduce_idx][0]]
                noise = temp_mask * adv_noise
                normalized_grad = noise
            else:
                abs_noise = torch.abs(adv_noise).view(1, -1)
                reduce_mask = abs_noise != 0
                abs_noise[abs_noise == 0] = 3

                reduce_num = torch.sum(reduce_mask).data.clone().item()
                if reduce_num == 1:
                    break

                noise_show, noise_sort_idx = torch.sort(abs_noise)
                noise_sort_idx = noise_sort_idx.view(-1)
                noise_idx = noise_sort_idx[reduce_idx]
                reduce_mask[0, noise_idx] = 0
                temp_mask = reduce_mask.view(1, 3, int(im_size), int(im_size))
                noise = temp_mask * adv_noise
                normalized_grad = noise

            with torch.no_grad():
                alpha = 1
                a = [alpha + i * 2 / 1000 for i in range(0, 50)]
                a = np.asarray(a)
                search_num = len(a)

                ex_temp_eps = torch.from_numpy(a).view(-1, 1, 1, 1).float().cuda()
                ex_normalized_grad = normalized_grad
                ex_scaled_grad = ex_normalized_grad.mul(ex_temp_eps)

                ex_real_A = real_A.repeat(int(search_num), 1, 1, 1)
                ex_temp_A = ex_real_A - ex_scaled_grad
                ex_temp_A = clip(ex_temp_A, ex_real_A, eps)
                ex_adv = torch.clamp(ex_temp_A, 0, 1)

                ex_temp_A = Variable(ex_adv.data, requires_grad=True)
                ex_logist_B = netT(ex_temp_A)
                _, pre = torch.max(ex_logist_B, 1)

                comp = torch.eq(target, pre)
                top1 = torch.sum(comp).float() / pre.size()[0]
                found = False

                if top1 != 0:  # exists at least one adversarial sample
                    for i in range(search_num):
                        if comp[i] == 1:
                            logits = netT(ex_temp_A[i:i + 1])
                            pre = torch.argmax(logits, 1)
                            if pre == target:
                                adv = ex_temp_A[i:i + 1]
                                found = True
                                break
                reduce_idx += 1
                if not found:
                    count += 1
                else:
                    if group:
                        order.append(importance_dict[reduce_idx - 1][0])
                    count = 0
                if count > interval_num and group:
                    break
        order.sort()
        for i in range(len(order)):
            adv_mask = adv_mask[torch.arange(adv_mask.size(0)) != order[i] - i]
    return adv


def perturbation_importance(real_A, adv, netT, target, adv_mask, mask, segments=None):
    advimportance = pretrained_G(adv)

    advimportance = advimportance * adv_mask
    importance_dict = dict()
    importance_mean = advimportance.mean(dim=1).mean(dim=1).mean(dim=1)
    length = importance_mean.size()[0]
    for i in range(length):
        importance_dict[i] = importance_mean[i].item()
    importance_dict = sorted(importance_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    decrease_num = int(importance_mean.size()[0] * 0.1)
    for i in range(decrease_num):
        mask -= adv_mask[importance_dict[i][0]]
    order = []
    for i in range(decrease_num):
        order.append(importance_dict[i][0])
    order.sort()
    for i in range(decrease_num):
        adv_mask = adv_mask[torch.arange(adv_mask.size(0)) != order[i] - i]
    return mask, adv_mask


def clip(adv_A, real_A, eps):
    g_x = real_A - adv_A
    clip_gx = torch.clamp(g_x, min=-eps, max=eps)
    adv_x = real_A - clip_gx
    return adv_x


def cal_ss(segments, index):
    segments = segments - index
    segments = torch.abs(segments)
    segments = torch.clamp(segments, 0, 1) - 1
    segments = torch.abs(segments).float()
    return segments


def superpixel(image, n_segments):
    tempImg = image.cpu().permute(0, 2, 3, 1).data
    segments = slic(tempImg[0], n_segments=n_segments)
    segments = torch.from_numpy(segments)
    segments = torch.unsqueeze(segments, dim=0)
    segments = segments.repeat(3, 1, 1)
    segments = torch.unsqueeze(segments, dim=0)
    return segments


def CWLoss(logits, target, kappa=0, sept=False, tar=True):
    target = torch.ones(logits.size(0)).to(device).mul(target.float())
    target_one_hot = Variable(torch.eye(1000)[target.long()].to(device))
    real = torch.sum(target_one_hot * logits, 1)
    other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)
    if tar:
        return torch.sum(torch.max(other - real, kappa))
    elif not sept:
        return torch.sum(torch.max(real - other, kappa))
    else:
        return torch.max(real - other, kappa)


def cal_group(real_A, adv, segments):
    adv_noise = real_A - adv
    adv_noise = adv_noise.cpu()
    cur_group = [0]
    for index, num in enumerate(n_segments):
        max_iter = torch.max(segments)
        for j in range(1, max_iter + 1):
            segmentstemp = cal_ss(segments, j)
            for k in range(3):
                sum = torch.norm(segmentstemp[0, k, :, :] * adv_noise[0, k, :, :], p=0)
                if sum != 0:
                    cur_group[index] += 1
    return cur_group[0]


def save_image(real_A, adv):
    path_real = './image/real.png'
    path_adv = './image/adv.png'
    path_noise = './image/noise.png'

    adv_noise = real_A - adv
    torchvision.utils.save_image(real_A, path_real)
    torchvision.utils.save_image(adv, path_adv)
    torchvision.utils.save_image(adv_noise * 255, path_noise, normalize=True)


main()
