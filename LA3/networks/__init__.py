import torch

from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
# from torchvision import models
import numpy as np

from LA3.networks.resnet import ResNet
from LA3.networks.pyramidnet import PyramidNet
from LA3.networks.shakeshake.shake_resnet import ShakeResNet
from LA3.networks.wideresnet import WideResNet, WideResNet_Binary
from LA3.networks.shakeshake.shake_resnext import ShakeResNeXt
from LA3.networks.efficientnet_pytorch import EfficientNet, RoutingFn

from LA3.networks.neural_predictor_model import Neural_Predictor_Model, Neural_Predictor_Embedding_Model, Neural_Predictor_Embedding_Class_Model, Neural_Predictor_Embedding_Class_Op_Model, Neural_Predictor_Embedding_Op_Model, Neural_Predictor_Embedding_Class_Op_Pair_Model



def get_model(conf, num_class=10, local_rank=-1):
    name = conf['type']

    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet40_2_binary':
        model = WideResNet_Binary(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10_binary':
        model = WideResNet_Binary(28, 10, dropout_rate=0.0, num_classes=num_class)

    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_class)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_class)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)

    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)

    elif name == 'pyramid':
        model = PyramidNet('cifar10', depth=conf['depth'], alpha=conf['alpha'], num_classes=num_class, bottleneck=conf['bottleneck'])

    elif name == 'SwinTransformer':
        from transformers import SwinConfig, SwinForImageClassification
        config = SwinConfig()
        model = SwinForImageClassification(config)

    elif 'efficientnet' in name:
        model = EfficientNet.from_name(name, condconv_num_expert=conf['condconv_num_expert'], norm_layer=None)  # TpuBatchNormalization
        if local_rank >= 0:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        def kernel_initializer(module):
            def get_fan_in_out(module):
                num_input_fmaps = module.weight.size(1)
                num_output_fmaps = module.weight.size(0)
                receptive_field_size = 1
                if module.weight.dim() > 2:
                    receptive_field_size = module.weight[0][0].numel()
                fan_in = num_input_fmaps * receptive_field_size
                fan_out = num_output_fmaps * receptive_field_size
                return fan_in, fan_out

            if isinstance(module, torch.nn.Conv2d):
                # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L58
                fan_in, fan_out = get_fan_in_out(module)
                torch.nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, val=0.)
            elif isinstance(module, RoutingFn):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, val=0.)
            elif isinstance(module, torch.nn.Linear):
                # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L82
                fan_in, fan_out = get_fan_in_out(module)
                delta = 1.0 / np.sqrt(fan_out)
                torch.nn.init.uniform_(module.weight, a=-delta, b=delta)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, val=0.)
        model.apply(kernel_initializer)
    else:
        raise NameError('no model named, %s' % name)

    if local_rank >= 0:
        device = torch.device('cuda', local_rank)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        if torch.cuda.is_available():
            model = model.cuda()
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)

    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'cifar10_class': 10,
        'cifar10_class_prob': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'reduced_cifar10_4000': 10,
        'cifar100': 100,
        'reduced_cifar100': 100,
        'cifar100_class': 100,
        'reduced_cifar100_class_op': 100,
        'cifar10_class_op': 10,
        'cifar100_class_op': 100,
        'cifar100_class_prob': 100,
        'cifar100_class_filter': 100,
        'reduced_cifar100_4000': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'reduced_svhn_4000': 10,
        'imagenet': 1000,
        'imagenet_op': 1000,
        'reduced_imagenet': 1000,
        'reduced_imagenet_all_cls': 1000,
        'reduced_imagenet_all_cls_mem': 1000,
        'exin_img': 18,
        'omniglot': 1623,
        'reduced_omniglot': 1623,
    }[dataset]



def get_neural_predictor_class_embedding_op_model(local_rank=-1, num_class=100):
    model = Neural_Predictor_Embedding_Class_Op_Model(num_cls=num_class)

    if local_rank >= 0:
        device = torch.device('cuda', local_rank)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        if torch.cuda.is_available():
            model = model.cuda()
    #         model = DataParallel(model)

    cudnn.benchmark = True
    return model


def get_neural_predictor_class_embedding_op_pair_model(local_rank=-1, num_class=100):
    model = Neural_Predictor_Embedding_Class_Op_Pair_Model(num_cls=num_class)

    if local_rank >= 0:
        device = torch.device('cuda', local_rank)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        if torch.cuda.is_available():
            model = model.cuda()
    #         model = DataParallel(model)

    cudnn.benchmark = True
    return model
