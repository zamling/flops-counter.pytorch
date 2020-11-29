import argparse
import sys

import torch
import torchvision.models as models

from ptflops import  get_model_info

pt_models = {'resnet18': models.resnet18,
             'resnet50': models.resnet50,
             'alexnet': models.alexnet,
             'vgg16': models.vgg16,
             'squeezenet': models.squeezenet1_0,
             'densenet': models.densenet161,
             'inception': models.inception_v3}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='resnet18')
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--out_dir',type=str,default='./output.txt')
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    net = pt_models[args.model]()

    if torch.cuda.is_available():
        net.cuda(device=args.device)

    flops_model = get_model_info(net, (3, 224, 224),ost=ost)
    print('the total flops are: ',flops_model.get_total_flops())
    print('the total params are',flops_model.get_total_param())
    print('the flops of layer1 in Resnet18',flops_model.get_layer_flops(name='layer1'))
    print('the params of layer1 in Resnet18',flops_model.get_layer_params(name='layer1'))
    print('record the total results in %s'%(args.out_dir))
    flops_model.output_info_to_file(args.out_dir)

