import torch.nn as nn
import torch

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.conv1 = self.get_conv(3, 64, n_convolutions=2)
        self.conv2 = self.get_conv(64, 128, n_convolutions=2)
        self.conv3 = self.get_conv(128, 256, n_convolutions=3)
        self.conv4 = self.get_conv(256, 512, n_convolutions=3)
        self.conv5 = self.get_conv(512, 512, n_convolutions=3)

        self.deconv5 = self.get_deconv(512, 512, n_convolutions=3)
        self.deconv4 = self.get_deconv(512, 256, n_convolutions=3)
        self.deconv3 = self.get_deconv(256, 128, n_convolutions=3)
        self.deconv2 = self.get_deconv(128, 64, n_convolutions=2)
        self.deconv1 = self.get_deconv(64, 1, n_convolutions=2, not_last=False)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.upsampling = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def get_conv(self, inp_filt, out_filt, n_convolutions):
        conv = nn.Sequential(
            nn.Conv2d(inp_filt, out_filt, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_filt),
            nn.ReLU(),
            nn.Conv2d(out_filt, out_filt, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_filt),
            nn.ReLU()
        )
        if n_convolutions == 3:
            conv.add_module('conv', nn.Conv2d(out_filt, out_filt, kernel_size=(3, 3), padding=(1, 1)))
            conv.add_module('batch', nn.BatchNorm2d(out_filt))
            conv.add_module('relu', nn.ReLU())
        return conv

    def get_deconv(self, inp_filt, out_filt, n_convolutions, not_last=True):
        if n_convolutions == 3:
            out_n = inp_filt
        elif n_convolutions == 2:
            out_n = out_filt
        conv = nn.Sequential(
            nn.Conv2d(inp_filt, inp_filt, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(inp_filt),
            nn.ReLU(),
            nn.Conv2d(inp_filt, out_n, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_n),
            nn.ReLU()
        )
        if n_convolutions == 3:
            conv.add_module('conv', nn.Conv2d(inp_filt, out_filt, kernel_size=(3, 3), padding=(1, 1)))
            conv.add_module('batch', nn.BatchNorm2d(out_filt))
            if not_last:
                conv.add_module('relu', nn.ReLU())
        return conv

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))

        x = self.upsampling(self.deconv5(x))
        x = self.upsampling(self.deconv4(x))
        x = self.upsampling(self.deconv3(x))
        x = self.upsampling(self.deconv2(x))
        x = self.upsampling(self.deconv1(x))
        return self.sigmoid(x)


class UNet(nn.Module):
    def __init__(self, mode='pool'):
        super(UNet, self).__init__()

        if mode not in ['pool', 'stride', 'dilate']:
            raise ValueError("Invalid mode. Accepted values for mode are only 'pool', 'stride' and 'dilate'")

        self.mode = mode

        self.conv1 = self.get_conv(3, 64)
        self.conv2 = self.get_conv(64, 128)
        self.conv3 = self.get_conv(128, 256)
        self.conv4 = self.get_conv(256, 512)

        if mode == 'pool':
            self.pool = nn.ModuleList([nn.MaxPool2d(kernel_size=(2, 2)) for i in range(4)])
        elif mode == 'stride':
            self.pool = nn.ModuleList(
                [nn.Conv2d(N, N, kernel_size=(3, 3), stride=2, padding=(1, 1)) for N in [64, 128, 256, 512]])
        elif mode == 'dilate':
            self.pool = nn.ModuleList(
                [nn.Conv2d(N, N, kernel_size=(2, 2), dilation=256 * 32 // N) for N in [64, 128, 256, 512]])

        self.bottle = self.get_conv(512, 1024, deconv=True)

        self.deconv4 = self.get_conv(1024, 512, deconv=True)
        self.deconv3 = self.get_conv(512, 256, deconv=True)
        self.deconv2 = self.get_conv(256, 128, deconv=True)
        self.deconv1 = self.get_conv(128, 64)

        self.last = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )

    def get_conv(self, inp_filt, out_filt, deconv=False):
        conv = nn.Sequential(
            nn.Conv2d(inp_filt, out_filt, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_filt),
            nn.ReLU(),
            nn.Conv2d(out_filt, out_filt, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_filt),
            nn.ReLU()
        )
        if deconv:
            if self.mode == 'pool':
                conv.add_module('upsampling', nn.UpsamplingNearest2d(scale_factor=2))
                conv.add_module('conv', nn.Conv2d(out_filt, out_filt // 2, kernel_size=(3, 3), padding=(1, 1)))
            elif self.mode == 'stride':
                conv.add_module('strided_conv', nn.ConvTranspose2d(out_filt, out_filt // 2,
                                                                   kernel_size=(2, 2), stride=2))
            elif self.mode == 'dilate':
                conv.add_module('dalated_conv', nn.ConvTranspose2d(out_filt, out_filt // 2,
                                                                   kernel_size=(2, 2),
                                                                   dilation=1024 * 16 // out_filt))

        return conv

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool[0](x1))  # 256 -> 128
        x3 = self.conv3(self.pool[1](x2))  # 128 -> 64
        x4 = self.conv4(self.pool[2](x3))  # 64 -> 32

        x_out = self.bottle(self.pool[3](x4))  # 32 -> 32
        x_out = torch.concat([x4, x_out], dim=1)

        x_out = self.deconv4(x_out)  # 32 -> 64
        x_out = torch.concat([x3, x_out], dim=1)

        x_out = self.deconv3(x_out)  # 64 -> 128
        x_out = torch.concat([x2, x_out], dim=1)

        x_out = self.deconv2(x_out)  # 128 -> 256
        x_out = torch.concat([x1, x_out], dim=1)

        return self.last(self.deconv1(x_out))