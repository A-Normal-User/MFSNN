import torch
import torch.nn as nn
num_group = 53

class Residual_2_2_gelu(nn.Module):
    def __init__(self, input_View):
        super().__init__()
        self.L1 = nn.Sequential(
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            )
        self.L2 = nn.Sequential(
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            )
    def forward(self, X):
        X = self.L1(X)
        Y = self.L2(X)
        return Y + X
# 神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 分支1：处理分子结构向量
        self.fc1 = nn.Sequential(
            nn.Linear(num_group, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            )
        # 分支2：处理温度向量
        self.fc3 = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            )
        # 深度学习层
        self.fc_U = nn.Sequential(
            Residual_2_2_gelu(144),
            Residual_2_2_gelu(144),
            Residual_2_2_gelu(144),
            nn.GELU(),
            nn.Linear(144, 1)
            )
        self.fc_alphaij = nn.Sequential(
            Residual_2_2_gelu(144),
            Residual_2_2_gelu(144),
            nn.GELU(),
            nn.Linear(144, 1)
            )
    def forward(self, x):
        # 分离输入
        x1 = torch.log(x[:, :53] + 1)  # 分子结构向量1
        x2 = torch.log(x[:, 53:53*2] + 1)  # 分子结构向量2
        x3 = x[:, 53*2:] # 温度项
        # 分支1
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        x3 = torch.abs(self.fc3(x3 / 256))
        temp12 = torch.cat((x1, x2, x3), dim=1)
        temp21 = torch.cat((x2, x1, x3), dim=1)
        # 拟合NRTL的U_ij参数
        U12 = self.fc_U(temp12)
        U11 = self.fc_U(torch.cat((x1, x1, x3), dim=1))
        U21 = self.fc_U(temp21)
        U22 = self.fc_U(torch.cat((x2, x2, x3), dim=1))
        # 拟合NRTL的alpha_ij参数
        alphaij = self.fc_alphaij(temp12) + self.fc_alphaij(temp21)
        return torch.cat((U12 - U22, U21 - U11, alphaij), dim=1)