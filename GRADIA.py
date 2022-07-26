import argparse
import cv2
import os
import numpy as np
import json

import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from os import path
import random

import time

f = open('reasonablity_val.json')
reasonablity_labels = json.load(f)
f.close()

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_iou(x, y):
    intersection = np.bitwise_and(x,y)
    union = np.bitwise_or(x,y)

    iou = np.sum(intersection)/np.sum(union)

    return iou

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def show_cam_on_image(img, mask, path, file_name):
    save_path = os.path.join(path, file_name)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_BONE)
    heatmap = np.float32(heatmap) / 255

    # uint_img = np.uint8(mask * 255)
    # heatmap = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

    # cam = heatmap + np.float32(img)
    # cam = np.float32(img)
    cam = np.multiply(heatmap, np.float32(img))

    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))

def resize_attention_label(path_to_attn, width = 7, height = 7):
    path_to_attn_resized = {}
    for img_path, img_att in path_to_attn.items():
        att_map = np.uint8(img_att)*255

        # cv2.imwrite('test_in.png', att_map)
        # 7x7 can be too small to show attention details, blur it to reward near by pixels
        img_att_resized = cv2.resize(att_map, (width,height), interpolation=cv2.INTER_AREA)

        img_att_resized = cv2.GaussianBlur(img_att_resized, (3, 3), 0)
        # cv2.imwrite('test_out.png', img_att_resized)
        path_to_attn_resized[img_path]=np.float32(img_att_resized/np.max(img_att_resized))
    return path_to_attn_resized

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def generate_path_to_attentions(reasonablity = True):
    path_to_attn = {}

    f = open('attention_val.json')
    valid_att = json.load(f)
    f.close()

    for item in valid_att:
        _, img_path = os.path.split(item['img'])
        attention = item['attention']
        path_to_attn[img_path] = np.array(attention)

    f = open('attention_val_update.json')
    valid_att_update_list = json.load(f)
    f.close()

    for item in valid_att_update_list:
        _, img_path = os.path.split(item['img'])
        attention = item['attention']
        path_to_attn[img_path] = np.array(attention)

    f = open('attention_val_remove.json')
    valid_att_remove_list = json.load(f)
    f.close()

    for item in valid_att_remove_list:
        _, img_path = os.path.split(item['img'])
        path_to_attn.pop(img_path)


    if reasonablity:
       # print('Number of attention maps before selection:', len(path_to_attn))
       # remove (pop) those img that the model already got 'reasonable accurate' during phase 1 learning
       for img_path, label in reasonablity_labels.items():
           if label == 'Reasonable Accurate' and img_path in path_to_attn.keys():
               path_to_attn.pop(img_path)
       #     # setting 1
       #     if label == 'Unreasonable Accurate' and img_path in path_to_attn.keys():
       #         path_to_attn.pop(img_path)
       #     # Setting 2
       #     # if label == 'Reasonable Inaccurate' and img_path in path_to_attn.keys():
       #     #     path_to_attn.pop(img_path)
       #     # Setting 3

       # # random way
       # path_to_attn = dict(random.sample(path_to_attn.items(), 600))

    # print('Number of attention maps for val set:', len(path_to_attn))

    f = open('attention_test.json')
    test_att = json.load(f)
    f.close()

    # add test label to the dict
    for item in test_att:
        _, img_path = os.path.split(item['img'])
        attention = item['attention']
        path_to_attn[img_path] = np.array(attention)

    # print('Final total number of attention maps:', len(path_to_attn))

    return path_to_attn

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class ImageFolderWithMaps(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithMaps, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)

        # Add pred_weight & att_weight here
        if tail in path_to_attn_resized:
            true_attention_map = path_to_attn_resized[tail]
        else:
            true_attention_map = np.zeros((7,7), dtype=np.float32)

        tuple_with_map = (original_tuple + (true_attention_map,))
        return tuple_with_map

class ImageFolderWithMapsAndWeights(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithMapsAndWeights, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)

        # default weights
        pred_weight = 1
        att_weight = 0
        # Add pred_weight & att_weight here
        if tail in path_to_attn_resized:
            true_attention_map = path_to_attn_resized[tail]

            if tail in reasonablity_labels:
                label = reasonablity_labels[tail]

                if label == 'Unreasonable Inaccurate':
                    pred_weight = 1
                    att_weight = 1
                    # print('Find Unreasonable Inaccurate for image:', path)

                elif label == 'Unreasonable Accurate':
                    pred_weight = 0.4
                    att_weight = 1.6
                    # print('Find Unreasonable Accurate for image:', path)

                elif label == 'Reasonable Inaccurate':
                    pred_weight = 0.8
                    att_weight = 0.2
                    # print('Find Reasonable Inaccurate for image:', path)
                # else:
                # print('Error: No matches for image:', path)
        else:
            true_attention_map = np.zeros((7,7), dtype=np.float32)


        tuple_with_map_and_weights = (original_tuple + (true_attention_map, pred_weight, att_weight))
        return tuple_with_map_and_weights

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='Number of epoch to run')
    parser.add_argument('--data_dir', default='gender_data', type=str)
    parser.add_argument('--model_dir', type=str, default='./model_save/',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--model_name', type=str, default='',
                        help='The model filename that will be used for evaluation or phase 2 fine-tuning.')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--trainWithMap', dest='trainWithMap', action='store_true',
                        help='train with edited attention map')
    parser.add_argument('--reasonablity', dest='reasonablity', action='store_true',
                        help='Whether to fine-tune model in phase 2 using reasonablity matrix sample selection')
    parser.add_argument('--attention_weight', default=1.0, type=float,
                        help='Scale factor that balance between task loss and attention loss')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    # if args.use_cuda:
    #     print("Using GPU for acceleration")
    # else:
    #     print("Using CPU for computation")

    return args

args = get_args()
path_to_attn = generate_path_to_attentions(args.reasonablity)
# resize attention label from 224x224 to 14x14
path_to_attn_resized = resize_attention_label(path_to_attn)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        # self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def get_attention_map(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]

        target = features[-1].squeeze()

        weights = torch.mean(grads_val, axis=(2, 3)).squeeze()

        if self.cuda:
            cam = torch.zeros(target.shape[1:]).cuda()
        else:
            cam = torch.zeros(target.shape[1:])

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        # cam = torch.sigmoid(10*(cam-0.5))

        return cam, output

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            # print('model is looking at class', index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # cam = 1 / (1 + np.exp(-10 * (cam - 0.5)))

        return cam

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

# def attention_shift_study(data_loader):
#     print('Attention shift study with resize 7x7.')
#     for batch_idx, (inputs, targets, paths) in enumerate(data_loader):
#         for img_path in paths:
#             _, img_fn = os.path.split(img_path)
#             img = cv2.imread(img_path, 1)
#             img = np.float32(cv2.resize(img, (224, 224))) / 255
#
#             if img_fn in path_to_attn:
#                 mask_orignal = path_to_attn[img_fn]
#                 # show_cam_on_image(img, mask_orignal, 'attention_resize', img_path)
#
#                 mask_resize = path_to_attn_resized[img_fn]
#                 mask_resize_back = cv2.resize(mask_resize, (224, 224))
#
#                 img = cv2.hconcat([img, img])
#                 mask = np.concatenate((mask_orignal, mask_resize_back), axis=1)
#                 show_cam_on_image(img, mask, 'attention_resize_smooth_new', img_path)

def model_test(model, test_loader, output_attention= False, output_iou = False, output_reasonbality = None):
    print('start testing')
    # model.eval()
    iou = AverageMeter()
    ious = {}
    st = time.time()
    outputs_all = []
    targets_all = []
    img_fns = []

    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)
    y_true = np.array([])
    y_pred = np.array([])
    misclassified = np.array([])

    for batch_idx, (inputs, targets, paths) in enumerate(test_loader):
        y_true = np.append(y_true, targets)
        misclassified = np.append(misclassified, paths)

        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            y_pred = np.append(y_pred, predicted)

        if output_attention:
            for img_path in paths:
                _, img_fn = os.path.split(img_path)

                img_fns.append(img_fn)

                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input)
                show_cam_on_image(img, mask, 'attention', img_path)

                if output_iou and img_fn in path_to_attn:
                    item_att_binary = (mask > 0.75)
                    # cv2.imwrite('model_att.png', np.uint8(255 * item_att_binary.astype(np.float)))

                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    # cv2.imwrite('target_att.png', np.uint8(255 * target_att_binary.astype(np.float)))

                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    # print('iou:', single_iou)
                    iou.update(single_iou.item(), 1)
                    ious[img_fn]=single_iou.item()

        outputs_all += outputs
        targets_all += targets

    et = time.time()
    test_time = et - st

    misclassified_img = []
    # get misclassified img set
    for idx in range(len(misclassified)):
        if y_true[idx] != y_pred[idx]:
            _, img_path = os.path.split(misclassified[idx])
            misclassified_img.append(img_path)

    # save the filename as list
    with open('img_fns.json', 'w') as fp:
        json.dump(img_fns, fp)

    with open('misclassified.json', 'w') as fp:
        json.dump(misclassified_img, fp)

    with open('IOU.json', 'w') as fp:
        json.dump(ious, fp)

    test_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()

    if output_reasonbality is not None:
        reasonbality={}
        reasonbality_matrix =np.zeros((2,2))

        f = open(output_reasonbality)
        file = json.load(f)
        f.close()

        for item in file:
            _, img_path = os.path.split(item['img'])
            att_acc = item['label']

            if img_path == '':
                continue
            # Four cases
            if att_acc == 'unreasonable':
                if img_path in misclassified_img:
                    # case [1,1]: Unreasonable Inaccurate
                    reasonbality[img_path]='Unreasonable Inaccurate'
                else:
                    # case [0,1]: Unreasonable Accurate
                    reasonbality[img_path]='Unreasonable Accurate'
            else:
                if img_path in misclassified_img:
                    # case [1,0]: Reasonable Inaccurate
                    reasonbality[img_path]='Reasonable Inaccurate'
                else:
                    # case [0,0]: Reasonable Accurate
                    reasonbality[img_path]='Reasonable Accurate'

        # generate the reasonablity matrix
        for label in reasonbality.values():
            if label == 'Unreasonable Inaccurate':
                reasonbality_matrix[1, 1] += 1
            elif label == 'Unreasonable Accurate':
                reasonbality_matrix[0, 1] += 1
            elif label == 'Reasonable Inaccurate':
                reasonbality_matrix[1, 0] += 1
            elif label == 'Reasonable Accurate':
                reasonbality_matrix[0, 0] += 1
            else:
                print('Unknown label detected:', label)

        # print the reasonablity matrix
        print(reasonbality_matrix)
        # save the reasonablity results
        with open('reasonablity_gradia_test.json', 'w') as fp:
            json.dump(reasonbality, fp)

    return test_acc, iou.avg


def model_train_with_map(model, train_loader, val_loader):
    ####################################################################################################
    # phase 2: fine-tune the model on validation set
    ####################################################################################################
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    # attention_criterion = nn.MSELoss()
    attention_criterion = nn.L1Loss(reduction='none')
    # attention_criterion = nn.KLDivLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    best_val_iou = 0
    best_val_acc = 0

    # load grad_cam module
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)
    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets, target_maps, pred_weight, att_weight) in enumerate(train_loader):
            attention_loss = 0
            if args.use_cuda:
                inputs, targets, target_maps, pred_weight, att_weight= inputs.cuda(), targets.cuda(non_blocking=True), target_maps.cuda(), pred_weight.cuda(), att_weight.cuda()
            att_maps = []
            att_map_labels = []
            att_weights = []
            outputs = model(inputs)

            for input, target, target_map, valid_weight in zip(inputs, targets, target_maps, att_weight):
                # only train on img with attention labels
                if valid_weight > 0.0:
                    # get attention maps from grad-CAM
                    att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target)
                    att_maps.append(att_map)
                    att_map_labels.append(target_map)
                    att_weights.append(valid_weight)

            # compute losses
            task_loss = task_criterion(outputs, targets)
            task_loss = torch.mean(pred_weight * task_loss)

            if att_maps:
                att_maps = torch.stack(att_maps)
                att_map_labels = torch.stack(att_map_labels)
                att_weights = torch.stack(att_weights)
                attention_loss += attention_criterion(att_maps, att_map_labels)
                attention_loss = torch.mean(att_weights * torch.mean(torch.mean(attention_loss, dim=-1), dim=-1))

                # this exp trick is for gradually decrease the effect of attention label during the course of learning
                # change below line to default by: loss = task_loss + args.attention_weight * attention_loss
                # loss = task_loss + (args.attention_weight * 0.95 ** epoch) * attention_loss

                loss = task_loss + attention_loss
            else:
                loss = task_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all += outputs
            targets_all += targets

            # print('Batch_idx :', batch_idx, ', task_loss', task_loss, ', attention_loss', attention_loss)

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()

        '''
            Valid
        '''
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []

        iou = AverageMeter()
        for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            with torch.no_grad():
                outputs = model(inputs)

            for img_path in paths:
                _, img_fn = os.path.split(img_path)
                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input)
                # show_cam_on_image(img, mask, 'attention', img_path)

                if img_fn in path_to_attn:
                    item_att_binary = (mask > 0.75)
                    # cv2.imwrite('model_att.png', np.uint8(255 * item_att_binary.astype(np.float)))

                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    # cv2.imwrite('target_att.png', np.uint8(255 * target_att_binary.astype(np.float)))

                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    # print('iou:', single_iou)
                    iou.update(single_iou.item(), 1)

            outputs_all += outputs
            targets_all += targets

        et = time.time()
        test_time = et - st

        val_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()
        val_iou = iou.avg
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model, os.path.join(args.model_dir, 'model_out_' + str(epoch)))
            print('UPDATE!!!')
        else:
            torch.save(model, os.path.join(args.model_dir, 'model_out_' + str(epoch)))
        # print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc)
        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc, 'Val IOU:', val_iou)

    return best_val_iou

def model_train(model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val_acc = 0

    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all+=outputs
            targets_all+=targets

            print('Batch_idx :', batch_idx, ', loss', loss)

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()

        '''
            Valid
        '''
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)

                outputs_all += outputs
                targets_all += targets

        et = time.time()
        val_time = et - st

        val_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()

        # Calculate Valid AUC. Update the best model based on highest AUC score.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, os.path.join(args.model_dir, 'model_out'))
            print('UPDATE!!!')
        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc)

    return best_val_acc


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # path_to_attn = generate_path_to_attentions(args.reasonablity)
    # # resize attention label from 224x224 to 7x7
    # path_to_attn_resized = resize_attention_label(path_to_attn)

    task_criterion = nn.CrossEntropyLoss()
    attention_criterion = nn.MSELoss()

    # Data loading code
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    testdir = os.path.join(args.data_dir, 'test')
    # phase 2 training set, currently defined by the combination of phase 1's training set and the validation set
    p2traindir = os.path.join(args.data_dir, 'p2_train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    p2_train_loader_without_map = torch.utils.data.DataLoader(
        datasets.ImageFolder(p2traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # p2_train_loader = torch.utils.data.DataLoader(
    #     ImageFolderWithMaps(p2traindir, transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.train_batch, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    p2_train_loader = torch.utils.data.DataLoader(
        ImageFolderWithMapsAndWeights(p2traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = models.resnet50(pretrained=True)
    # replace the orignal output layer from 1000 classs to 2 class for man and woman task
    model.fc = nn.Linear(2048, 2)
    # if args.use_cuda:
    #     model.cuda()


    if args.evaluate:
        # attention_shift_study(val_loader)

        model = torch.load(os.path.join(args.model_dir, args.model_name))

        # evaluate model on validation set
        # val_acc, val_iou = model_test(model, val_loader, output_attention= True, output_iou = True)
        # print('Finish Testing. Val Acc:', val_acc, ', Val IOU:', val_iou)

        # evaluate model on test set, set output_attention to true if you want to save the model generated attention
        test_acc, test_iou = model_test(model, test_loader, output_attention=True, output_iou=True)

        # Notice that to get the reasonablity matrix, you also need human assessment such as in 'phase2_edit_attention_accuracy_test.json' file
        # test_acc, test_iou= model_test(model, test_loader, output_attention= True, output_iou = True, output_reasonbality = 'phase2_edit_attention_accuracy_test.json')
        print('Finish Testing. Test Acc:', test_acc, ', Test IOU:', test_iou)
    else:
        if args.trainWithMap:
            # phase 2: train the model on val+train set with additional attention labels
            model = torch.load(os.path.join(args.model_dir, args.model_name))

            # for phase 2 baseline without attention supervision
            # best_val_acc= model_train(model, p2_train_loader_without_map, val_loader)
            # print('Finish Training. Best Validation acc:', best_val_acc)

            best_val_iou = model_train_with_map(model, p2_train_loader, test_loader)
            print('Finish Training. Best Validation IOU:', best_val_iou)
        else:
            # phase 1: train the model on training set
            best_val_acc= model_train(model, train_loader, val_loader)
            print('Finish Training. Best Validation acc:', best_val_acc)
