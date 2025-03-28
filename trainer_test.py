import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from trainer import GRPPOTrainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HardDecision(nn.Module):
    def forward(self, x):
        return (x > 0).float()

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.front_net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x, noise_std=0.5):
        global device
        for layer in self.front_net:
            if isinstance(layer, nn.Linear):
                # 临时噪声，不修改原始参数
                noisy_weight = layer.weight + torch.randn_like(layer.weight) * noise_std
                noisy_bias = layer.bias + torch.randn_like(layer.bias) * noise_std
                x = torch.nn.functional.linear(x, noisy_weight, noisy_bias)
            else:
                x = layer(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.decision = HardDecision()
        self.back_net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x, noise_std=0.1):
        global device
        x = self.decision(x)
        x = self.back_net(x)
        """for layer in self.back_net:
            if isinstance(layer, nn.Linear):
                # 临时噪声，不修改原始参数
                noisy_weight = layer.weight + torch.randn_like(layer.weight) * noise_std
                noisy_bias = layer.bias + torch.randn_like(layer.bias) * noise_std
                x = torch.nn.functional.linear(x, noisy_weight, noisy_bias)
            else:
                x = layer(x)"""
        return x

# --- 2. 定义数据集 ---
class SignDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.rand(size, 1) * 2 - 1  # 生成[-1, 1]的随机数
        self.labels = (self.data > 0).float()    # 正数为1，负数为0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    创建一个DataLoader来批量加载SignDataset数据
    
    参数:
        dataset: SignDataset实例
        batch_size: 每个batch的大小，默认32
        shuffle: 是否随机打乱数据，默认True
    
    返回:
        DataLoader对象
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 可以根据需要调整并行加载的工作进程数
        drop_last=False  # 是否丢弃最后一个不完整的batch
    )
    return dataloader
# --- 3. 定义奖励函数 ---

# --- 4. 配置训练参数 ---
class GRPOConfig:
    def __init__(self):
        self.batch_size = 8
        self.epochs = 5
        self.num_generations = 5  # 每个输入生成的样本数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_weights=None
        self.log_interval = 10
        self.max_grad_norm = 0.01

# --- 5. 初始化模型和数据 ---
net1 = Net1()
net2 = Net2()
train_dataset = SignDataset(size=1000)
config = GRPOConfig()

# --- 6. 训练 ---
if __name__ == "__main__":
    # 初始化训练器
    trainer = GRPPOTrainer(
        model_parts=[net1, net2],
        reward_funcs=[lambda samples,target:(-nn.functional.mse_loss(samples, target.expand_as(samples),reduction='none'))],
        train_dataset=train_dataset,
        DataLoader=create_dataloader,
        optimizers=(torch.optim.SGD, 0.001),
        args=config
    )
    
    # 开始训练
    model=trainer.train()
    while True:
        try:
            user_input = input("请输入一个数字进行推理（或输入 'exit' 退出）：")
            if user_input.lower() == 'exit':
                print("退出推理模式。")
                break
            user_input = float(user_input)
            input_tensor = torch.tensor([[user_input]], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
            print(f"模型输出：{output.item():.4f}")
        except ValueError:
            print("输入无效，请输入一个数字。")
        except KeyboardInterrupt:
            print("\n退出推理模式。")
            break