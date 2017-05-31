import sys
import os
import math
import mxnet as mx
import memonger

mx_workspace = 8000
classes = 19
feat_stride = 8


def get_symbol():

    from symbol.symbol import cfg as symcfg
    symcfg['lr_type'] = 'alex'
    symcfg['workspace'] = mx_workspace
    symcfg['bn_use_global_stats'] = True

    from symbol.resnet_v2 import fcrna_model_a1, rna_model_a1
    net = fcrna_model_a1(classes, feat_stride, bootstrapping=False)
    # net = rna_model_a1(classes)
    return net

batch_size = 24
net = get_symbol()
dshape = (batch_size, 3, 500, 500)
net_mem_planned = memonger.search_plan(net, data=dshape)
new_cost = memonger.get_cost(net_mem_planned, data=dshape)
old_cost = memonger.get_cost(net, data=dshape)
os.environ['MXNET_BACKWARD_DO_MIRROR'] = '1'
old_cost2 = memonger.get_cost(net, data=dshape)

print('Naive feature map cost=%d MB' % old_cost)
print('Best feature map cost=%d MB' % new_cost)
print('Mirror feature map cost=%d MB' % old_cost2)
# You can savely feed the net to the subsequent mxnet training script.
