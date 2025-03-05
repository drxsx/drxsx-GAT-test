import torch
import torch.nn as nn

# 硬决策模块（不可导）
class HardDecision(nn.Module):
    def forward(self, x):
        return (x>0).float()  # 输出0或1，阻断梯度

# 决策前网络 (N1)
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.front_net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x, noise_std=0.1):
        # 遍历网络中的每一层并添加噪声
        fnet=self.front_net
        for layer in self.front_net:
            if isinstance(layer, nn.Linear):  # 只有在是nn.Linear层时才添加噪声
                # 添加噪声到权重和偏置
                noise_weight = torch.randn_like(layer.weight) * noise_std
                noise_bias = torch.randn_like(layer.bias) * noise_std

                # 在每次前向传播时修改权重和偏置
                layer.weight.data += noise_weight
                layer.bias.data += noise_bias
        x=self.front_net(x)
        self.front_net=fnet
        return x 
    
    def sample_outputs(self, x, num_samples=5, noise_std=0.1):
        samples = []
        for _ in range(num_samples):
            samples.append(self.forward(x))
        
        samples = torch.stack(samples)  # 将所有样本堆叠成一个批次
        return samples


# 决策后网络 (N2)
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.decision = HardDecision()
        self.back_net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        x = self.decision(x)
        x = self.back_net(x)
        return x

# 组合模型用于推理
class CombinedModel(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
    
    def forward(self, x):
        beta = self.net1(x)
        gamma = self.net2(beta)
        return gamma

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = Net1().to(device)
model2 = Net2().to(device)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
criterion = nn.MSELoss()


# 修改后的训练循环，添加 KL 散度正则化
inputs = torch.rand(3000, 1) * 4 - 2  # 范围 [-2, 2]
targets = (inputs > 0).float()  # 二值目标
inputs, targets = inputs.to(device), targets.to(device)

# 训练循环
batch_size = 10  # 每个batch 10个样本
num_samples = 5

for epoch in range(200):
    # 使用batch_size分批处理数据
    for i in range(0, len(inputs), batch_size):
        # 取一个batch的数据
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        # 获取旧策略输出
        beta_old = model1(batch_inputs).detach()  # 旧策略输出
        optimizer1.zero_grad()
        beta_samples = model1.sample_outputs(batch_inputs, num_samples)
        
        losses = []
        log_probs = []
        print(beta_samples,batch_inputs)
        for i in range(num_samples):
            beta_i = beta_samples[i]
            gamma_i = model2(beta_i)
            loss_i = criterion(gamma_i, batch_targets)
            losses.append(loss_i)
            # 计算更精确的 log_prob
            log_prob = -0.5 * ((beta_i - model1(batch_inputs)) ** 2) / (0.1 ** 2)
            log_probs.append(log_prob.mean())

        losses = torch.stack(losses)
        log_probs = torch.stack(log_probs)
        baseline = losses.mean()
        advantages = -losses + baseline

        # 添加 KL 散度正则化
        kl_div = ((beta_samples - beta_old) ** 2).mean()  # 简化的 KL 近似
        loss_n1 = -(advantages * log_probs).mean() + 0.01 * kl_div

        loss_n1.backward()
        optimizer1.step()

        # 更新 Net2
        optimizer2.zero_grad()
        beta = model1(batch_inputs).detach()
        gamma = model2(beta)
        loss_n2 = criterion(gamma, batch_targets)
        loss_n2.backward()
        optimizer2.step()
    print(f"Epoch {epoch}, Loss: {loss_n2.item():.4f}")
# 创建推理模型
model = CombinedModel(model1, model2).to(device)
model.eval()

# 推理部分
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
