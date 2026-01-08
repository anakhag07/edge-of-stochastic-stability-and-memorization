import torch as T
import torch
import torch.nn as nn

from utils.resnet_new import resnet14, resnet20, ResNet
from utils.resnet_bn import resnet10 as resnet10_bn, ResNet as ResNetBN
import torch.nn.functional as F

from einops import rearrange, repeat

from utils.data import get_dataset_presets
from typing import Union
from pathlib import Path
from contextlib import contextmanager



def get_model_presets():
    model_presets = {
        'linear': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2
            }
        },
        'linear_s': {
            'type': 'linear',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1,
                'bias': True
            }
        },
        'linear_l': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4
            }
        },
        'lin_tiny': {
            'type': 'linear',
            'params': {
                'hidden_dim': 2,
                'n_layers': 1
            }
        },
        'mlp': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2
            }
        },
        'mlp2': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 2
            }
        },
        'mlp3': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 3
            }
        },
        'mlp_s': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1
            }
        },
        'mlp_l': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4
            }
        },
        'cnn': {
            'type': 'cnn',
            'params': {
                'hidden_dim': 512,
            }
        },
        'resnet': {
            'type': 'resnet',
            'params': {},
        },
        'resnet_bn': {
            'type': 'resnet_bn',
            'params': {},
        }
    }
    return model_presets





class SquaredLoss(nn.modules.loss._Loss):
    '''
    Basically MSE, but doesn't average over the dimensions.
    With added support for sampling_vector (aka weighting of the samples, aka mask) the samples! 
    Used to do GD with noise to simulate SGD
    '''
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 ) -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: T.Tensor, target: T.Tensor,
                sampling_vector: T.Tensor = None,
                reduction: str = None
                ) -> T.Tensor:
        '''
        THE MASK NEEDS TO BE "NORMALIZED" - i.e. with expected value of 1/n*unit_vector, 
        and NOT just unit_vector, without the normalization
        this is because without mask there is averaging happening!
        Inherently, this is to perform Jacobian-vector products
        '''

        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape for the loss to operate as expected.\nDid you forget to squeeze the output?")
        

        if sampling_vector is not None:
            total_L2 = F.mse_loss(input, target, reduction='none') 
            
            if len(target.shape) != 1:
                loss_per_sample = total_L2.sum(dim=-1) # shape = (batch_size,)
            else:
                loss_per_sample = total_L2

            assert len(loss_per_sample.shape) == 1
            sampled_loss = T.dot(loss_per_sample, sampling_vector) # L \dot \omega - where \omega is the sampling vector
            return sampled_loss
        
        # if len(target.shape) != 1:
        #     # used to 
        #     multiplier = input.size(-1)
        # else:
        #     multiplier = 1.
        
        total_L2 =  F.mse_loss(input, target, reduction='none') # shape = (batch_size, num_classes)
        if len(target.shape) != 1:
            loss_per_sample = total_L2.sum(dim=-1) # shape = (batch_size,)
        else:
            loss_per_sample = total_L2

        
        if not reduction is None:
            if reduction == 'none':
                return loss_per_sample

            raise ValueError(f"Are you sure you want to use reduction={reduction}? Double-check what you doing - maybe use self.reduction variable at __init__ instead?\n")
        
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        if self.reduction == 'sum':
            return loss_per_sample.sum()
        # we are not introducing this just as a safety
        # if self.reduction == 'none':
        #     return loss_per_sample
        
        raise ValueError("Unknown reduction type")
        



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def __repr__(self):
        return f"MLP({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim})"


class CNN(nn.Module):
    def __init__(self, fc_hidden_dim, output_dim):
        super(CNN, self).__init__()
        self.fc_hidden_dim = fc_hidden_dim
        # self.conv1 = nn.Conv2d(3, 64, 3, 1)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.convs = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1), # 64*30*30
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), # 64*28*28
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 64, 14

                nn.Conv2d(64, 128, 3, 1), # 128, 12
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 128, 6
        )
        self.fcs = nn.Sequential(
                nn.Linear(128*6*6, fc_hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, output_dim, bias=True)
        )
        # self.fc1 = nn.Linear(128*6*6, width, bias=False)
        # self.fc2 = nn.Linear(width, 1, bias=False)
        # self.apply(_weights_init)

    def forward(self, x):
        x = self.convs(x)
        x = rearrange(x, 'b c w h -> b (c w h)')
        x = self.fcs(x)
        return x


class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, bias=True):
        super(Linear, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"Linear({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim})"


def prepare_net(model_type: str,
                params: dict
                ):
    if model_type == 'linear':
        net = Linear(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'], params['bias'])
    
    if model_type == 'mlp':
        net = MLP(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'])
    
    if model_type == 'cnn':
        net = CNN(params['hidden_dim'], params['output_dim'])
    
    if model_type == 'resnet':
        net = resnet14()
    
    if model_type == 'resnet_bn':
        raise "Not implemented - you are still using old resnet_bn"
        net = resnet10_bn()

    return net

def prepare_net_dataset_specific(model_name: str,
                                 dataset: str,
                                 ):
    '''
    Returns the model specific to the procided dataset
    '''
    model_presets = get_model_presets()
    params = model_presets[model_name]['params']
    model_type = model_presets[model_name]['type']

    dataset_presets = get_dataset_presets()
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']

    net = prepare_net(model_type, params)

    return net

    

def initialize_mlp(net, scale=None):
    if scale is None:
        scale=1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data = m.weight.data * scale
            nn.init.zeros_(m.bias)


def initialize_cnn(net, scale=None):
    if scale is None:
        scale = 1.0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def initialize_resnet_old(net, scale=0.01):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data *= 0.1
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    net.fc.weight.data *= scale

def initialize_resnet(net, scale=None):
    if scale is None:
        scale = 0.01
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, torch.nn.Linear):
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    T.nn.init.normal_(net.linear.weight, std=scale)


def initialize_resnet_bn(net, scale=0.1):
    for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    # initialize fc layer
    nn.init.kaiming_normal_(net.fc.weight, mode='fan_out', nonlinearity='relu')
    # self.fc.weight.data.mul_(0.1)
    nn.init.constant_(net.fc.bias, 0)

    # # Zero-initialize the last BN in each residual branch,
    # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    # for m in net.modules():
    #     if isinstance(m, BasicBlock):
    #         pass
    #         nn.init.constant_(m.bn2.weight, 0)

    # custom scale
    net.fc.weight.data *= scale


def initialize_linear(net, scale=None):
    if scale is None:
        scale=1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data = m.weight.data * scale
            nn.init.zeros_(m.bias)


@contextmanager
def temp_seed(seed):
    '''
    Temporarily sets the seed for the random number generator
    This is a context
    '''
    if seed is None:
        yield
        return
        
    state = T.get_rng_state()
    T.manual_seed(seed)
    if T.cuda.is_available():
        cuda_state = T.cuda.get_rng_state()
        T.cuda.manual_seed(seed)
    
    try:
        yield
    finally:
        T.set_rng_state(state)
        if T.cuda.is_available():
            T.cuda.set_rng_state(cuda_state)


def initialize_net(net, scale=None, seed=None):

    with temp_seed(seed):
        if isinstance(net, Linear):
            initialize_linear(net, scale=scale)
        elif isinstance(net, MLP):
            initialize_mlp(net, scale=scale)    
        elif isinstance(net, ResNet):
            initialize_resnet(net, scale=scale)
        elif isinstance(net, ResNetBN):
            initialize_resnet_bn(net, scale=scale)
        elif isinstance(net, CNN):
            initialize_cnn(net, scale=scale)
        else:
            raise ValueError("Unknown net type")

def prepare_optimizer(net, lr, momentum, adam, weight_decay: float = 0.0):
    if adam:
        if momentum is not None:
            raise ValueError(
                "Momentum is not supported for Adam. "
                "Change the code if you need to change the params in Adam"
            )

        if weight_decay and weight_decay > 0:
            # Only use AdamW when weight decay is requested
            return T.optim.AdamW(
                net.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )

        # Default:
        return T.optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
        )

    # SGD branch (weight_decay here is classic L2)
    if momentum is not None:
        return T.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    return T.optim.SGD(net.parameters(), lr=lr, momentum=0, weight_decay=weight_decay)



def get_path_of_last_net(path: Union[str, Path], not_final=False):
    path = Path(path)
    if path.is_dir():
        files = list(path.glob('*.pt'))
        if 'net_final.pt' in [file.name for file in files]:
            return path / 'net_final.pt'
        if len(files) == 0:
            return None
        files.sort(key=lambda x: x.stat().st_mtime)

        return files[-1]
    else:
        return path
