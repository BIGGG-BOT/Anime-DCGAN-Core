import torch
import torch.nn as nn
from torchvision import transforms
from create_dataset import My_dataset, save_img
from torch.utils.data import DataLoader
# from model512 import Generator, Discriminator
from model64 import Generator, Discriminator

# 图像变换
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 网络设置图片大小为 512*512
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = My_dataset(r'D:\桌面\1', transform=transform)   # 数据集位置
batch_size, epochs = 32, 500
my_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

discriminator = Discriminator()
generator = Generator()

if torch.cuda.is_available():

    discriminator = discriminator.cuda()
    generator = generator.cuda()


d_optimizer = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.99), lr=1e-4)  # betas为adam算法两个动量参数
g_optimizer = torch.optim.Adam(generator.parameters(), betas=(0.5, 0.99), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(epochs):

    for i, img in enumerate(my_dataloader):

        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        real_img = img.cuda()
        fake_img = generator(noise)

        real_label = torch.ones(batch_size).cuda()
        fake_label = torch.zeros(batch_size).cuda()
        real_out = discriminator(real_img)
        fake_out = discriminator(fake_img)
        real_loss = criterion(real_out, real_label)
        fake_loss = criterion(fake_out, fake_label)

        d_loss = real_loss + fake_loss
        d_optimizer.zero_grad()

        d_loss.backward()
        d_optimizer.step()

        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake_img = generator(noise)
        output = discriminator(fake_img)

        g_loss = criterion(output, real_label)
        g_optimizer.zero_grad()

        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 5 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D_real: {:.6f},D_fake: {:.6f}'.format(
                epoch, epochs, d_loss.data.item(), g_loss.data.item(),
                real_out.data.mean(), fake_out.data.mean()  # 打印的是真实图片的损失均值
            ))
        if epoch == 0 and i == len(my_dataloader) - 1:          # 保存真实图像
            save_img(img[:64, :, :, :], './sample/real_images.png')
        if (epoch+1) % 50 == 0 and i == len(my_dataloader)-1:             # 每50个epoch保存一次预测图像
            save_img(fake_img[:64, :, :, :], './sample/fake_images_{}.png'.format(epoch + 1))

torch.save(generator.state_dict(), './generator.pth')        # 保存权重文件
torch.save(discriminator.state_dict(), './discriminator.pth')
