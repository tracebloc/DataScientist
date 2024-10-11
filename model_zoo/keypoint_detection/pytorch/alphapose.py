import torch.nn as nn
import torch.nn.functional as F

# Configuration
framework = "pytorch"
model_type = ""
main_class = "MyModel"
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_feature_points = 16


config = {
        'PRESET': {
            'NUM_JOINTS': 16
        },
        'NUM_DECONV_FILTERS': [256, 128, 64],
        'NUM_LAYERS': 50
    }

class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        # if not inspect.isclass(module_class):
        #     raise TypeError('module must be a class, but got {}'.format(
        #         type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


SPPE = Registry('sppe')



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, norm_layer=nn.BatchNorm2d, dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = dcn is not None

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
            self.with_modulated_dcn = dcn.get('MODULATED', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        # else:
        #     from .dcn import DeformConv, ModulatedDeformConv
        #     self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
        #     if not self.with_modulated_dcn:
        #         conv_op = DeformConv
        #         offset_channels = 18
        #     else:
        #         conv_op = ModulatedDeformConv
        #         offset_channels = 27

        #     self.conv2_offset = nn.Conv2d(
        #         planes,
        #         self.deformable_groups * offset_channels,
        #         kernel_size=3,
        #         stride=stride,
        #         padding=1)
        #     self.conv2 = conv_op(
        #         planes,
        #         planes,
        #         kernel_size=3,
        #         stride=stride,
        #         padding=1,
        #         deformable_groups=self.deformable_groups,
        #         bias=False)

        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.with_dcn:
            out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = F.relu(self.bn2(self.conv2(out, offset, mask)))
        else:
            offset = self.conv2_offset(out)
            out = F.relu(self.bn2(self.conv2(out, offset)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """ ResNet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d, dcn=None, stage_with_dcn=(False, False, False, False)):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        self.architecture = architecture
        # assert architecture in ["resnet18", "resnet34", "resnet50", "resnet101", 'resnet152']
        layers = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
        }
        self.inplanes = 64
        if architecture == "resnet18" or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_dcn = [dcn if with_dcn else None for with_dcn in stage_with_dcn]

        self.layer1 = self.make_layer(
            self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        if self.architecture == "resnet18" or self.architecture == 'resnet34':
            layers.append(block(self.inplanes, planes, stride, downsample,
                                norm_layer=self._norm_layer))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes,
                                    norm_layer=self._norm_layer))

        else:
            layers.append(block(self.inplanes, planes, stride, downsample,
                                norm_layer=self._norm_layer, dcn=dcn))

            self.inplanes = planes * block.expansion

            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes,
                                    norm_layer=self._norm_layer))

        return nn.Sequential(*layers)


@SPPE.register_module
class MyModel(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, cfg=config):
        super(MyModel, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.deconv_dim = cfg['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm   # noqa: F401,F403
        # assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = getattr(tm, f"resnet{cfg['NUM_LAYERS']}")(pretrained=True)

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(
            self.deconv_dim[2], self._preset_cfg['NUM_JOINTS'], kernel_size=1, stride=1, padding=0)
        print(self._preset_cfg['NUM_JOINTS'])
        # Define a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer to predict keypoints' x, y coordinates and visibility
        num_features = self.deconv_dim[2]
        self.fc = nn.Linear(16, self._preset_cfg['NUM_JOINTS'] * 3)

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            2048, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                # if self.deconv_with_bias:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # logger.info('=> init {}.weight as 1'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.preact(x)
        out = self.deconv_layers(out)
        out = self.final_layer(out)
        out = self.global_avg_pool(out)
        print("Shape before fc layer:", out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(-1, self._preset_cfg['NUM_JOINTS'], 3)
        return out