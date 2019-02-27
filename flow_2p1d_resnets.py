import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


class FlowLayer(nn.Module):

    def __init__(self, channels=1, params=[1,1,1,1,1], n_iter=10):
        super(FlowLayer, self).__init__()
        self.n_iter = n_iter
        if params[0]:
            self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,1,1,1))
            self.img_grad2 = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).transpose(3,2).repeat(channels,1,1,1))
        else:
            self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.img_grad2 = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).transpose(3,2).repeat(channels,1,1,1), requires_grad=False)            

            
        if params[1]:
            self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1))
            self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1))
            self.div = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1))
            self.div2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1))            
        else:
            self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.div = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.div2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1), requires_grad=False)

            
        self.channels = channels
        
        self.t = 0.3
        self.l = 0.15
        self.a = 0.25        

        if params[2]:
            self.t = nn.Parameter(torch.FloatTensor([self.t]))
        if params[3]:
            self.l = nn.Parameter(torch.FloatTensor([self.l]))
        if params[4]:
            self.a = nn.Parameter(torch.FloatTensor([self.a]))


    def norm_img(self, x):
        mx = torch.max(x)
        mn = torch.min(x)
        x = 255*(x-mn)/(mn-mx)
        return x
            
    def forward_grad(self, x):
        grad_x = F.conv2d(F.pad(x, (0,1,0,0)), self.f_grad, groups=self.channels)
        grad_x[:,:,:,-1] = 0
        
        grad_y = F.conv2d(F.pad(x, (0,0,0,1)), self.f_grad2, groups=self.channels)
        grad_y[:,:,-1,:] = 0
        return grad_x, grad_y


    def divergence(self, x, y):
        tx = F.pad(x[:,:,:,:-1], (1,0,0,0))
        ty = F.pad(y[:,:,:-1,:], (0,0,1,0))
        
        grad_x = F.conv2d(F.pad(tx, (0,1,0,0)), self.div, groups=self.channels)
        grad_y = F.conv2d(F.pad(ty, (0,0,0,1)), self.div2, groups=self.channels)
        return grad_x + grad_y
        
        
    def forward(self, x, y):
        u1 = torch.zeros_like(x)
        u2 = torch.zeros_like(x)
        l_t = self.l * self.t
        taut = self.a/self.t

        grad2_x = F.conv2d(F.pad(y,(1,1,0,0)), self.img_grad, groups=self.channels, padding=0, stride=1)
        grad2_x[:,:,:,0] = 0.5 * (x[:,:,:,1] - x[:,:,:,0])
        grad2_x[:,:,:,-1] = 0.5 * (x[:,:,:,-1] - x[:,:,:,-2])

        
        grad2_y = F.conv2d(F.pad(y, (0,0,1,1)), self.img_grad2, groups=self.channels, padding=0, stride=1)
        grad2_y[:,:,0,:] = 0.5 * (x[:,:,1,:] - x[:,:,0,:])
        grad2_y[:,:,-1,:] = 0.5 * (x[:,:,-1,:] - x[:,:,-2,:])
        
        p11 = torch.zeros_like(x.data)
        p12 = torch.zeros_like(x.data)
        p21 = torch.zeros_like(x.data)
        p22 = torch.zeros_like(x.data)

        gsqx = grad2_x**2
        gsqy = grad2_y**2
        grad = gsqx + gsqy + 1e-12

        rho_c = y - grad2_x * u1 - grad2_y * u2 - x
        
        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

            v1 = torch.zeros_like(x.data)
            v2 = torch.zeros_like(x.data)
            mask1 = (rho < -l_t*grad).detach()
            v1[mask1] = (l_t * grad2_x)[mask1]
            v2[mask1] = (l_t * grad2_y)[mask1]
            
            mask2 = (rho > l_t*grad).detach()
            v1[mask2] = (-l_t * grad2_x)[mask2]
            v2[mask2] = (-l_t * grad2_y)[mask2]

            mask3 = ((mask1^1) & (mask2^1) & (grad > 1e-12)).detach()
            v1[mask3] = ((-rho/grad) * grad2_x)[mask3]
            v2[mask3] = ((-rho/grad) * grad2_y)[mask3]
            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2

            u1 = v1 + self.t * self.divergence(p11, p12)
            u2 = v2 + self.t * self.divergence(p21, p22)
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)
            
            p11 = (p11 + taut * u1x) / (1. + taut * torch.sqrt(u1x**2 + u1y**2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * torch.sqrt(u1x**2 + u1y**2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * torch.sqrt(u2x**2 + u2y**2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * torch.sqrt(u2x**2 + u2y**2 + 1e-12))
            del u1x
            del u1y
            del u2x
            del u2y
            
        return u1, u2


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



def conv3x3(in_planes, out_planes, stride=1,T=1):
    """ 3D convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[T,3,3], stride=stride,
                     padding=(1 if (T>1 or stride>1) else 0, 1,1), bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,T=3)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=[3,1,1], bias=False, padding=(1,0,0))
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=[1,3,3], stride=stride,
                               padding=((1 if stride > 1 else 0),1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=150, in_channels=3, input_size=112, input_len=16, n_iter=3, learnable=[1,1,1,1,1], config=0, dropout=0.3):
        self.inplanes = 64
        self.in_channels = in_channels
        super(ResNet, self).__init__()

        self.config = config
        self.flow_cmp = nn.Conv3d(128*block.expansion, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.flow_layer = FlowLayer(channels=32, n_iter=n_iter, params=learnable)
        self.flow_conv = nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)

        # Flow-of-flow
        self.flow_cmp2 = nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.flow_layer2 = FlowLayer(channels=32, n_iter=n_iter, params=learnable)
        self.flow_conv2 = nn.Conv3d(64, 128*block.expansion, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bnf = nn.BatchNorm3d(128*block.expansion)
        
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        size = int(math.ceil(input_size/32))
        length = int(math.ceil(input_len/16))
        self.avgpool = nn.AvgPool3d((length,size,size), stride=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        torch.nn.init.zeros_(self.bnf.weight)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, padding=((1 if stride > 1 else 0),0,0), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x is BxCxTxHxW
        b,c,t,h,w = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        _,ci,t,h,w = x.size()
        res = F.avg_pool3d(x,(3,1,1),1,0)#x[:,:,1:-1].contiguous()
        x = self.flow_cmp(x)
        x = self.flow_layer.norm_img(x)
        # handle time
        _,c,t,h,w = x.size()
        t = t-1
        # compute flow for 0,1,...,T-1
        #        and       1,2,...,T
        u,v = self.flow_layer(x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w), x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w))
        x = torch.cat([u,v], dim=1)

        x = x.view(b,t,c*2,h,w).permute(0,2,1,3,4).contiguous()
        x = self.flow_conv(x)
        
        # Flow-of-flow
        _,ci,t,h,w = x.size()
        x = self.flow_cmp2(x)
        x = self.flow_layer.norm_img(x)
        # handle time
        _,c,t,h,w = x.size()
        t = t-1
        # compute flow for 0,1,...,T-1
        #        and       1,2,...,T
        u,v = self.flow_layer2(x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w), x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w))
        x = torch.cat([u,v], dim=1)
        x = x.view(b,t,c*2,h,w).permute(0,2,1,3,4).contiguous()
        x = self.flow_conv2(x)
        x = self.bnf(x)
        x = x+res
        x = self.relu(x)



        x = self.layer3(x)
        x = self.layer4(x)

        # average pool BxC'xT'xH'xW' to be BxC
        # mean-pooling over T, H, and W
        # may want to look into this for better fully-conv support (i.e., per-frame classification)
        # or (temporal) pool after classifying
        x = self.avgpool(x)
        x = torch.mean(x, dim=2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        # return BxClasses

        return x

    def load_2d_state_dict(self, state_dict, strict=True):
        # slightly hacky trick to inflate 2d kernel to 3d by repeating over time axis
        state_dict = {k:v for k,v in state_dict.items() if 'fc' not in k}
        md = self.state_dict()
        for k,v in state_dict.items():
            if 'conv' in k or 'downsample.0' in k:
                if isinstance(v, nn.Parameter):
                    v = v.data
                if self.in_channels != 3 and k == 'conv1.weight':
                    v = torch.mean(v, dim=1).unsqueeze(1).repeat(1, self.in_channels, 1, 1)
                # CxKxHxW -> CxKxTxHxW
                D = md[k].size(2)
                v = v.unsqueeze(2).repeat(1,1,D,1,1)
                state_dict[k] = v

        md.update(state_dict)
        super(ResNet, self).load_state_dict(md, strict)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_2d_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, mode='rgb', **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_2d_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, mode='rgb', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if mode == 'rgb':
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=2, **kwargs)
    if pretrained:
        model.load_2d_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_2d_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_2d_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    # test resnet 50
    import torch
    d = torch.device('cuda')
    net = nn.DataParallel(resnet34(pretrained=True, mode='rgb'))
    net.to(d)

    vid = torch.rand((12,3,16,112,112)).to(d)

    print(net(vid).size())
