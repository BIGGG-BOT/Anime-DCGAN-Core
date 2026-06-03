from torch import nn


class Generator(nn.Module):

    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # ConvTranspose2d   out_size = (in_size - 1) * S + K - 2P + output_padding
            # outputpadding为输出特征图边缘需要填充的行（列）数，默认为0
            # out_shape = (1-1)*1-2*0+4 = 4*4
            nn.ConvTranspose2d(noise_dim, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # out_shape = (4-1)*2-2*1+4 = 8*8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # out_shape = (8-1)*2-2*1+4 = 16*16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # out_shape = (16-1)*2-2*1+4 = 32*32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # out_shape = (32-1)*2-2*1+4 = 64*64
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # out_shape = (64-1)*2-2*1+4 = 128*128
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # out_shape = (128-1)*2-2*1+4 = 256*256
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # out_shape = (128-1)*2-2*1+4 = 512*512
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):

        output = self.net(input)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # out_shape = 4*256*256
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # out_shape = 8*128*128
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # out_shape = 16*64*64
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # out_shape = 32*32*32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # out_shape = 64*16*16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # out_shape = 128*8*8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # out_shape = 256*4*4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4*4*256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):

        output = self.net(input)
        return output.view(-1)