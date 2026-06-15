import os
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from create_dataset import My_dataset, save_img
from torch.utils.data import DataLoader
from model64 import Generator, Discriminator

# 1. 物理隔离输出目录
os.makedirs('./sample_task2', exist_ok=True)

# 2. 图像变换 (缝合了 Task 1 的翻转增强)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5), # 核心：无损扩充数据
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. 加载 Task 2 离线清洗后的锐化数据集
dataset = My_dataset('./data/1_enhanced', transform=transform)   
batch_size, epochs = 32, 500
my_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

discriminator = Discriminator()
generator = Generator()

if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()

d_optimizer = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.99), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), betas=(0.5, 0.99), lr=1e-4)
criterion = nn.BCELoss()

# 4. 初始化 Loss 记录列表 (Task 3 要求)
G_losses = []
D_losses = []

print("🚀 开始执行 Task 2 标准流程训练...")
for epoch in range(epochs):
    for i, img in enumerate(my_dataloader):
        
        # =======================
        #  训练判别器 (Discriminator)
        # =======================
        noise = torch.randn(batch_size, 100, 1, 1).cuda() if torch.cuda.is_available() else torch.randn(batch_size, 100, 1, 1)
        real_img = img.cuda() if torch.cuda.is_available() else img
        fake_img = generator(noise)

        real_label = torch.ones(batch_size).cuda() if torch.cuda.is_available() else torch.ones(batch_size)
        fake_label = torch.zeros(batch_size).cuda() if torch.cuda.is_available() else torch.zeros(batch_size)
        
        real_out = discriminator(real_img)
        # 核心修复：添加 .detach() 截断计算图，防止误伤生成器且节约显存
        fake_out = discriminator(fake_img.detach()) 
        
        real_loss = criterion(real_out, real_label)
        fake_loss = criterion(fake_out, fake_label)
        d_loss = real_loss + fake_loss
        
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # =======================
        #  训练生成器 (Generator)
        # =======================
        noise = torch.randn(batch_size, 100, 1, 1).cuda() if torch.cuda.is_available() else torch.randn(batch_size, 100, 1, 1)
        fake_img = generator(noise)
        output = discriminator(fake_img)

        g_loss = criterion(output, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # 5. 记录 Loss
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if (i + 1) % 5 == 0:
            print('Epoch[{}/{}], d_loss:{:.6f}, g_loss:{:.6f} D_real: {:.6f}, D_fake: {:.6f}'.format(
                epoch, epochs, d_loss.data.item(), g_loss.data.item(),
                real_out.data.mean(), fake_out.data.mean()  
            ))
            
        # 6. 标准化输出路径
        if epoch == 0 and i == len(my_dataloader) - 1:          
            save_img(img[:64, :, :, :], './sample_task2/real_images.png')
        if (epoch+1) % 50 == 0 and i == len(my_dataloader)-1:             
            save_img(fake_img[:64, :, :, :], './sample_task2/fake_images_{}.png'.format(epoch + 1))

# 7. 保存带有 task2 后缀的权重
torch.save(generator.state_dict(), './generator_task2.pth')        
torch.save(discriminator.state_dict(), './discriminator_task2.pth')

# 8. 绘制并保存 Loss 曲线
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training (Task 2)")
plt.plot(G_losses, label="G_Loss", color='blue', alpha=0.7)
plt.plot(D_losses, label="D_Loss", color='orange', alpha=0.7)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('./sample_task2/loss_curve.png', bbox_inches='tight')
print("✅ 训练完成！权重文件与 Loss 曲线图已成功保存至 ./sample_task2。")