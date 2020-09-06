# import torch
# import torch.nn as nn
# from torch.nn.parameter import Parameter

import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid import dygraph
from paddle.fluid.layers import reduce_mean as mean

# torch-like Layers class definition
class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, padding_size):
        super(ReflectionPad2d, self).__init__()
        self.padding_size = [padding_size for _ in range(4)]
    
    def forward(self, x):
        return layers.pad2d(input=x, paddings=self.padding_size, mode='reflect')

class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.relu(x))
            return x
        else:
            return fluid.layers.relu(x)

class Tanh(fluid.dygraph.Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return layers.tanh(x)

class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, alpha=0.2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.set_value(layers.leaky_relu(x, alpha=self.alpha))
            return x
        else:
            return layers.leaky_relu(x, alpha=self.alpha)

class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale, resample='NEAREST'):
        super(Upsample, self).__init__()
        self.scale = scale
        self.resample = resample

    def forward(self, x):
        return layers.image_resize(x, scale=self.scale, resample=self.resample)

# manual implementation of variance computation
# adapted from paddle v2.0.0a source code
def var(input, dim=None, keep_dim=False, unbiased=True, name=None):
    rank = len(input.shape)
    dims = dim if dim != None and dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in dims]
    inp_shape = input.shape
    mean = layers.reduce_mean(input, dim=dim, keep_dim=True, name=name)
    tmp = layers.reduce_mean((input - mean)**2, dim=dim, keep_dim=keep_dim, name=name)
    if unbiased:
        n = 1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    return tmp

# my own attempted implementation of variance computation
# def var(input, dim=None, keep_dim=False, unbiased=True, name=None):
#     out = np.var(input.numpy(), axis=tuple(dim), keepdims=keep_dim, ddof=1)
#     return dygraph.to_variable(out)


# Not Learnable bias_attr for InstanceNorm
# IN_bias = fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0), trainable=False)

# class ResnetGenerator(nn.Module):
class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        # DownBlock = []
        # DownBlock += [nn.ReflectionPad2d(3),
        #               nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
        #               nn.InstanceNorm2d(ngf),
        #               nn.ReLU(True)]
        DownBlock = []
        DownBlock += [
            ReflectionPad2d(3),
            dygraph.Conv2D(input_nc, ngf, 7, bias_attr=False),
            dygraph.InstanceNorm(ngf),
            ReLU(True)
        ]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            # DownBlock += [nn.ReflectionPad2d(1),
            #               nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
            #               nn.InstanceNorm2d(ngf * mult * 2),
            #               nn.ReLU(True)]
            DownBlock += [
                ReflectionPad2d(1),
                dygraph.Conv2D(ngf * mult, ngf * mult * 2, 3, stride=2, bias_attr=False),
                dygraph.InstanceNorm(ngf * mult * 2),
                ReLU(True)
            ]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # self.DownBlock = nn.Sequential(*DownBlock)
        self.DownBlock = dygraph.Sequential(*DownBlock)

        # Class Activation Map
        # self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        # self.relu = nn.ReLU(True)
        self.gap_fc = dygraph.Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = dygraph.Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = dygraph.Conv2D(ngf * mult * 2, ngf * mult, 1)
        self.relu = ReLU(True)

        # Gamma, Beta block
        if self.light:
            # FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
            #       nn.ReLU(True),
            #       nn.Linear(ngf * mult, ngf * mult, bias=False),
            #       nn.ReLU(True)]
            FC = [
                dygraph.Linear(ngf * mult, ngf * mult, bias_attr=False),
                ReLU(True),
                dygraph.Linear(ngf * mult, ngf * mult, bias_attr=False),
                ReLU(True)
            ]
        else:
            # FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
            #       nn.ReLU(True),
            #       nn.Linear(ngf * mult, ngf * mult, bias=False),
            #       nn.ReLU(True)]
            FC = [
                dygraph.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                ReLU(True),
                dygraph.Linear(ngf * mult, ngf * mult, bias_attr=False),
                ReLU(True)
            ]
        # self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        # self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.gamma = dygraph.Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = dygraph.Linear(ngf * mult, ngf * mult, bias_attr=False)

        # self.FC = nn.Sequential(*FC)
        self.FC = dygraph.Sequential(*FC)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
            #              nn.ReflectionPad2d(1),
            #              nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
            #              ILN(int(ngf * mult / 2)),
            #              nn.ReLU(True)]
            UpBlock2 += [
                Upsample(scale=2, resample='NEAREST'),
                ReflectionPad2d(1),
                dygraph.Conv2D(ngf * mult, int(ngf * mult / 2), 3, bias_attr=False),
                ILN(int(ngf * mult / 2)),
                ReLU(True)
            ]

        # UpBlock2 += [nn.ReflectionPad2d(3),
        #              nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
        #              nn.Tanh()]
        UpBlock2 += [
            ReflectionPad2d(3),
            dygraph.Conv2D(ngf, output_nc, 7, bias_attr=False),
            Tanh()
        ]

        # self.UpBlock2 = nn.Sequential(*UpBlock2)
        self.UpBlock2 = dygraph.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        # adaptive_avg_pool2d_1 = dygraph.Pool2D(pool_size=x.shape[-2:], pool_type='avg') # pool into 1x1 feature map
        # gap = adaptive_avg_pool2d_1(x)
        # print('x', x.shape)
        gap = layers.adaptive_pool2d(x, 1, pool_type='avg')
        # print('gap', gap.shape)
        gap_logit = self.gap_fc(layers.reshape(gap, shape=(x.shape[0], -1)))
        # print('gap_logit', gap_logit.shape)
        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = layers.reshape(gap_weight, shape=(1, -1))
        # print('gap_weight', gap_weight.shape)
        gap = x * layers.unsqueeze(layers.unsqueeze(gap_weight, 2), 3)
        # print('gap', gap.shape)

        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        # adaptive_max_pool2d_1 = dygraph.Pool2D(pool_size=x.shape[-2:], pool_type='max') # pool into 1x1 feature map
        # gmp = adaptive_max_pool2d_1(x)
        gmp = layers.adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(layers.reshape(gmp, shape=(x.shape[0], -1)))
        gmp_weight = self.gmp_fc.parameters()[0]
        gmp_weight = layers.reshape(gmp_weight, shape=(1, -1))
        gmp = x * layers.unsqueeze(layers.unsqueeze(gmp_weight, 2), 3)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        # x = torch.cat([gap, gmp], 1)
        # x = self.relu(self.conv1x1(x))
        cam_logit = layers.concat([gap_logit, gmp_logit], 1)
        x = layers.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        # heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            # x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            # x_ = self.FC(x_.view(x_.shape[0], -1))
            # adaptive_avg_pool2d_1 = dygraph.Pool2D(pool_size=x.shape[-2:], pool_type='avg')
            # x_ = adaptive_avg_pool2d_1(x)
            x_ = layers.adaptive_pool2d(x, 1, pool_type='avg')
            x_ = self.FC(layers.reshape(x_, shape=(x_.shape[0], -1)))
        else:
            # x_ = self.FC(x.view(x.shape[0], -1))
            x_ = self.FC(layers.reshape(x, shape=(x.shape[0], -1)))

        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        # conv_block += [nn.ReflectionPad2d(1),
        #                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
        #                nn.InstanceNorm2d(dim),
        #                nn.ReLU(True)]
        conv_block +=[
            ReflectionPad2d(1),
            dygraph.Conv2D(dim, dim, 3, bias_attr=use_bias),
            dygraph.InstanceNorm(dim),
            ReLU(True)
        ]

        # conv_block += [nn.ReflectionPad2d(1),
        #                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
        #                nn.InstanceNorm2d(dim)]
        conv_block += [
            ReflectionPad2d(1),
            dygraph.Conv2D(dim, dim, 3, bias_attr=use_bias),
            dygraph.InstanceNorm(dim)
        ]

        # self.conv_block = nn.Sequential(*conv_block)
        self.conv_block = dygraph.Sequential(*conv_block)


    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        # self.pad1 = nn.ReflectionPad2d(1)
        # self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        # self.norm1 = adaILN(dim)
        # self.relu1 = nn.ReLU(True)
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = dygraph.Conv2D(dim, dim, 3, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU(True)

        # self.pad2 = nn.ReflectionPad2d(1)
        # self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        # self.norm2 = adaILN(dim)
        self.pad2 = ReflectionPad2d(1)
        self.conv2 = dygraph.Conv2D(dim, dim, 3, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        # self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        # self.rho.data.fill_(0.9)
        self.rho = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=fluid.initializer.Constant(0.9))

    def forward(self, input, gamma, beta):
        # in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        # out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        # ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        # out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        # out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        # out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        in_mean = mean(input, dim=[2, 3], keep_dim=True)
        in_var = var(input, dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / layers.sqrt(in_var + self.eps)
        ln_mean = mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = var(input, dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / layers.sqrt(ln_var + self.eps)
        # rho_expand = layers.expand(self.rho, [input.shape[0], 1, 1, 1])
        # out = rho_expand * out_in + (1-rho_expand) * out_ln
        out = self.rho * out_in + (1 - self.rho) * out_ln
        out = out * layers.unsqueeze(layers.unsqueeze(gamma, 2), 3) + layers.unsqueeze(layers.unsqueeze(beta, 2), 3)
        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        # self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        # self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        # self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        # self.rho.data.fill_(0.0)
        # self.gamma.data.fill_(1.0)
        # self.beta.data.fill_(0.0)
        self.rho = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=fluid.initializer.Constant(0.0))
        self.gamma = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=fluid.initializer.Constant(1.0))
        self.beta = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=fluid.initializer.Constant(0.0))

    def forward(self, input):
        # in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        # out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        # ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        # out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        # out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        # out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        in_mean = mean(input, dim=[2, 3], keep_dim=True)
        in_var = var(input, dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / layers.sqrt(in_var + self.eps)
        ln_mean = mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = var(input, dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / layers.sqrt(ln_var + self.eps)
        # rho_expand = layers.expand(self.rho, [input.shape[0], 1, 1, 1])
        # out = rho_expand * out_in + (1-rho_expand) * out_ln
        # out = out * layers.expand(self.gamma, [input.shape[0], 1, 1, 1]) + layers.expand(self.beta, [input.shape[0], 1, 1, 1])
        out = self.rho * out_in + (1 - self.rho) * out_ln
        out = out * self.gamma + self.beta
        return out


class SpectralNorm(fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(SpectralNorm, self).__init__()
        self.spectral_norm = dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        # model = [nn.ReflectionPad2d(1),
        #          nn.utils.spectral_norm(
        #          nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
        #          nn.LeakyReLU(0.2, True)]
        model = [
            ReflectionPad2d(1),
            SpectralNorm(
                dygraph.Conv2D(input_nc, ndf, 4, stride=2), dim=1),
            LeakyReLU(0.2, True)
        ]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            # model += [nn.ReflectionPad2d(1),
            #           nn.utils.spectral_norm(
            #           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
            #           nn.LeakyReLU(0.2, True)]
            model += [
                ReflectionPad2d(1),
                SpectralNorm(
                    dygraph.Conv2D(ndf * mult, ndf * mult * 2, 4, stride=2), dim=1),
                LeakyReLU(0.2, True)
            ]

        mult = 2 ** (n_layers - 2 - 1)
        # model += [nn.ReflectionPad2d(1),
        #           nn.utils.spectral_norm(
        #           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
        #           nn.LeakyReLU(0.2, True)]
        model += [
            ReflectionPad2d(1),
            SpectralNorm(
                dygraph.Conv2D(ndf * mult, ndf * mult * 2, 4), dim=1),
            LeakyReLU(0.2, True)
        ]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        # self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        # self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        # self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        # self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.gap_fc = SpectralNorm(dygraph.Linear(ndf * mult, 1, bias_attr=False), dim=0)
        self.gmp_fc = SpectralNorm(dygraph.Linear(ndf * mult, 1, bias_attr=False), dim=0)
        self.conv1x1 = dygraph.Conv2D(ndf * mult * 2, ndf * mult, 1)
        self.leaky_relu = LeakyReLU(0.2, True)

        # self.pad = nn.ReflectionPad2d(1)
        # self.conv = nn.utils.spectral_norm(
        #     nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.pad = ReflectionPad2d(1)
        self.conv = SpectralNorm(dygraph.Conv2D(ndf * mult, 1, 4, bias_attr=False), dim=1)

        # self.model = nn.Sequential(*model)
        self.model = dygraph.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        # adaptive_avg_pool2d_1 = dygraph.Pool2D(pool_size=x.shape[-2:], pool_type='avg')
        # gap = adaptive_avg_pool2d_1(x)
        gap = layers.adaptive_pool2d(x, 1, pool_type='avg')
        gap_logit = self.gap_fc(layers.reshape(gap, shape=(x.shape[0], -1)))
        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = layers.reshape(gap_weight, shape=(1, -1))
        gap = x * layers.unsqueeze(layers.unsqueeze(gap_weight, 2), 3)

        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        # adaptive_max_pool2d_1 = dygraph.Pool2D(pool_size=x.shape[-2:], pool_type='max')
        # gmp = adaptive_max_pool2d_1(x)
        gmp = layers.adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(layers.reshape(gmp, shape=(x.shape[0], -1)))
        gmp_weight = self.gmp_fc.parameters()[0]
        gmp_weight = layers.reshape(gmp_weight, shape=(1, -1))
        gmp = x * layers.unsqueeze(layers.unsqueeze(gmp_weight, 2), 3)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        # x = torch.cat([gap, gmp], 1)
        # x = self.leaky_relu(self.conv1x1(x))
        cam_logit = layers.concat([gap_logit, gmp_logit], 1)
        x = layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        # heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        # if hasattr(module, 'rho'):
        #     # w = module.rho.data
        #     # w = w.clamp(self.clip_min, self.clip_max)
        #     # module.rho.data = w
        #     module.rho.set_value(layers.clamp(module.rho, min=self.clip_min, max=self.clip_max))
        for param in module.parameters():
            if param.name.startswith('ada_iln') or (param.name.startswith('iln') and param.name.endswith('w_0')):
                # print('clipped!')
                clipped_param = layers.clamp(param, min=self.clip_min, max=self.clip_max)
                param.set_value(clipped_param)

