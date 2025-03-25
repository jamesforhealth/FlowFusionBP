import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_residual_blocks=9):
        super(Generator, self).__init__()
        
        # 初始卷积层
        model = [nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                 nn.InstanceNorm1d(64),
                 nn.ReLU(inplace=True)]
        
        # 下采样
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv1d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm1d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # 残差块
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # 上采样
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose1d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm1d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # 输出层
        model += [nn.Conv1d(64, output_channels, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv1d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv1d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import get_data_loaders
import itertools

def train_cyclegan(optical_path, pressure_path, num_epochs=200, batch_size=64, lr=0.0002, beta1=0.5, beta2=0.999):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据加载器
    optical_loader, pressure_loader = get_data_loaders(optical_path, pressure_path, batch_size)

    # 初始化模型
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # 损失函数
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # 优化器
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, beta2))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, beta2))

    # 训练循环
    for epoch in range(num_epochs):
        for i, (optical, pressure) in enumerate(zip(optical_loader, pressure_loader)):
            real_A = optical.to(device)
            real_B = pressure.to(device)

            # 训练生成器
            optimizer_G.zero_grad()

            # 身份损失
            same_B = G_AB(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            same_A = G_BA(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN 损失
            fake_B = G_AB(real_A)
            pred_fake = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

            fake_A = G_BA(real_B)
            pred_fake = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

            # 循环一致性损失
            recovered_A = G_BA(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = G_AB(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # 总生成器损失
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizer_G.step()

            # 训练判别器 A
            optimizer_D_A.zero_grad()
            pred_real = D_A(real_A)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = D_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # 训练判别器 B
            optimizer_D_B.zero_grad()
            pred_real = D_B(real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = D_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss G: {loss_G.item():.4f}, Loss D_A: {loss_D_A.item():.4f}, Loss D_B: {loss_D_B.item():.4f}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(G_AB.state_dict(), f'G_AB_epoch_{epoch+1}.pth')
            torch.save(G_BA.state_dict(), f'G_BA_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    optical_path = 'preprocessed_data/optical_segments.npy'
    pressure_path = 'preprocessed_data/pressure_segments.npy'
    
    # 测试时使用这些参数
    max_samples = 10000  # 只使用前10000个样本
    window_size = 200    # 窗口大小
    stride = 100         # 步长
    batch_size = 32      # 批次大小
    
    # 使用更小的参数进行测试
    train_cyclegan(
        optical_path, 
        pressure_path,
        batch_size=32,  # 减小批次大小
        num_epochs=100  # 减少训练轮数
    )