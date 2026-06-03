import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 64*64*3), # 数据集图片大小为64*64，3通道
            nn.Tanh()  # 最后必须用tanh，把数据分布到（-1，1）之间
        )
    def forward(self, x):  # x表示长度为100的噪声输入
        img = self.main(x)
        img = img.view(-1,3,64,64)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(64*64*3, 1024),
            nn.LeakyReLU(), # x小于零是是一个很小的值不是0，x大于0是还是x
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,1),
            nn.Sigmoid() # 保证输出范围为（0，1）的概率
        )
    def forward(self, x):
        img = x.view(-1, 64*64*3)
        img = self.main(img)
        return img