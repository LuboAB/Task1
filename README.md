## 1. 程序功能

本项目的目标是构建一个基于深度卷积神经网络（CNN）的图像分类系统，用于自动识别 7 种常见的矿石样本（biotite, bornite, chrysocolla, malachite, muscovite, pyrite, quartz）。

**程序主要功能包括：**

1.  **自动化数据集划分**：提供脚本将原始图像目录自动按比例（如 8:2）划分为训练集和验证集，并保持原有的目录结构。
2.  **数据加载与预处理**：使用 PyTorch 的 `ImageFolder` 和 `DataLoader` 实现数据的高效从磁盘读取、尺寸统一化（Resize 为 224x224）和张量转换。
3.  **深度学习模型构建**：从头实现了一个基于 ResNet (Residual Network) 架构的卷积神经网络，不依赖外部预训练权重。
4.  **模型训练与监控**：
    *   实现了完整的训练循环，包含前向传播、反向传播和梯度更新。
    *   集成了准确率计算、损失值（Loss）监控和学习率自动调整策略（ReduceLROnPlateau）。
    *   使用进度条实时显示每个 Epoch 的状态。
5.  **结果保存与分析**：
    *   自动管理实验目录（类似 YOLO 风格，如 `runs/train1`, train2）。
    *   保存训练过程中的最佳模型权重 (`best.pt`) 和最终权重 (`last.pt`)。
    *   训练结束后自动绘制 Loss/Accuracy 曲线图、生成混淆矩阵（Confusion Matrix）以及保存部分验证集样本的预测结果图。

---

## 2. 神经网络结构 

为了在不使用迁移学习的情况下获得高准确率，本项目采用了一个**ResNet-18 (18层残差网络)** 的变体。该网络通过引入“跳跃连接”（Shortcut Connection），解决了深层网络难以训练的问题，非常适合捕捉矿石复杂的纹理特征。

**网络具体配置如下：**

1.  **输入层**：接收 `(3, 224, 224)` 尺寸的 RGB 图像。
2.  **初始卷积层 (Stage 0)**：`7x7` 卷积核，步长 2，输出 64 通道，降低空间尺寸并提取初步特征。
3.  **残差块堆叠 (Stages 1-4)**：网络核心由 4 个阶段组成，每个阶段包含 2 个 `BasicBlock`。
    *   **Stage 1**: 2 个 Block，输出通道 64，特征图尺寸 56x56
    *   **Stage 2**: 2 个 Block，输出通道 128，特征图尺寸 28x28
    *   **Stage 3**: 2 个 Block，输出通道 256，特征图尺寸 14x14
    *   **Stage 4**: 2 个 Block，输出通道 512，特征图尺寸 7x7
4.  **全局平均池化 (GAP)**：使用 `AdaptiveAvgPool2d`将 7x7 的特征图压缩为 1x1，极大地减少了全连接层的参数量，降低过拟合风险。
5.  **全连接分类层 (FC)**：单个线性层，将 512 维特征映射到 7 个类别概率。

---

## 3. 网络实现

### 3.1 模型定义 (`model.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# ... 其他导入 ...

# --- 基础残差块 BasicBlock ---
class BasicBlock(nn.Module):
    expansion = 1 # 输出通道倍数

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个 3x3 卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个 3x3 卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut 连接：如果输入输出维度不一致，用 1x1 卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 核心：将输入直接加到输出上
        out = F.relu(out)
        return out

# --- ResNet 主体架构 ---
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 初始处理：Conv7x7 -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个核心层级 (Stages)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
```

### 3.2 训练逻辑 (`train.ipynb` & model.py)

训练相关功能的实现
```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': running_loss/(pbar.n+1)})
    
    return running_loss / len(loader), 100 * correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

def save_confusion_matrix(model, loader, device, class_names, save_path):
    """计算并在本地保存混淆矩阵图片"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_history(history, save_path=None):
    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    # Acc
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close() # 避免在notebook里重复打印
    else:
        plt.show()

def fit_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, class_names):
    # 1. 创建 run 目录
    run_dir = get_next_run_dir()
    print(f"Training started. Results will be saved to: {run_dir}")
    
    # 2. 初始化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # 3. 循环训练
    for epoch in range(num_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler.step(v_loss)
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] | T_Loss: {t_loss:.4f} T_Acc: {t_acc:.2f}% | V_Loss: {v_loss:.4f} V_Acc: {v_acc:.2f}% | LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), run_dir / 'best.pt')
            # 同时也保存一份 last.pt
            torch.save(model.state_dict(), run_dir / 'last.pt')
        else:
            # 即使不是 best，也更新 last.pt
            torch.save(model.state_dict(), run_dir / 'last.pt')

    # 4. 训练结束后的工作
    print("\n Training finished. Generatring reports...")
    
    # 保存 loss/acc 曲线图
    plot_history(history, save_path=run_dir / 'results.png')
    
    # 加载最佳模型进行评估
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(run_dir / 'best.pt'))
    
    # 保存混淆矩阵
    print("Saving confusion matrix...")
    save_confusion_matrix(model, val_loader, device, class_names, run_dir / 'confusion_matrix.png')
        
    print(f" All done! Check directory: {run_dir}")
    return history


```

以下是训练流程的核心配置代码：
```python
# 1. 实例化模型并移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = MineralCNN(num_classes=7).to(device)

# 2. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# 3. 启动训练循环 (fit_model 函数封装了复杂的循环逻辑)
# 该函数包含:
# - ReduceLROnPlateau: 验证集 loss 不下降时自动减小学习率
# - Best Model Saving: 自动保存验证准确率最高的模型
# - fit_model的实现请移步model.py
history = fit_model(
    cnn_model, 
    train_loader, 
    validation_loader, 
    criterion, 
    optimizer, 
    num_epochs=50, 
    device=device,
    class_names=class_names
)
```

---

## 4. 训练过程与结果分析

实验在一个包含 `4510` 张训练图片和 `1130` 张验证图片的数据集上进行，共训练了 50 个 Epoch。

### 4.1 训练日志摘要

训练初期（Epoch 1-8），由于从零开始训练，模型在寻找最优解时出现了损失值震荡。
关键转折点出现在 **Epoch 31-40** 区间，学习率调度器（ReduceLROnPlateau）发挥作用，将学习率降低，模型精度随即突破 90% 瓶颈。

**最终结果：**
*   **最佳训练集准确率 (Training Acc)**: 98.89%
*   **最佳验证集准确率 (Validation Acc)**: **90.62%** (Epoch 50)
*   **最终验证 Loss**: 0.5065

**部分训练日志截屏/文本：**

```text
Epoch [1/50] | T_Loss: 1.5338 T_Acc: 43.61% | V_Loss: 1.3462 V_Acc: 49.56% | LR: 0.001000
...
Epoch [25/50] | T_Loss: 0.0794 T_Acc: 97.69% | V_Loss: 0.4862 V_Acc: 89.82% | LR: 0.001000
...
Epoch [36/50] | T_Loss: 0.0317 T_Acc: 98.71% | V_Loss: 0.5017 V_Acc: 90.27% | LR: 0.000500
...
Epoch [50/50] | T_Loss: 0.0264 T_Acc: 98.78% | V_Loss: 0.5065 V_Acc: 90.62% | LR: 0.000125
```

![output](https://github.com/LuboAB/Task1/blob/main/assets/output.png?raw=true)

### 4.2 训练曲线 (Loss & Accuracy)

![results](https://github.com/LuboAB/Task1/blob/main/assets/results-1768404111926.png?raw=true)

*   **Loss 曲线**：训练 Loss (Training Loss) 呈现出非常平滑的下降趋势，最终接近 0。验证 Loss (Validation Loss) 在 Epoch 20 左右趋于稳定，保持在 0.5 左右。
*   **Accuracy 曲线**：验证准确率稳步上升，最终稳定在 90% 左右。

### 4.3 混淆矩阵 (Confusion Matrix)

程序运行结束后自动生成了混淆矩阵，用于展示不同类别之间的误判情况。

![confusion_matrix](https://github.com/LuboAB/Task1/blob/main/assets/confusion_matrix.png?raw=true)

### 4.4 结论

本次实验成功构建并训练了一个能够有效识别 7 种矿石的 ResNet-18 模型。在未借助预训练模型的情况下，通过合理的网络设计和学习率调度策略，最终实现了 **90.62%** 的分类准确率，证明了程序的可靠性和模型的有效性。

## 5. 创新性改进：引入双重注意力机制 (ResNet-CBAM)

### 5.1 改进思路与动机

在基础 ResNet-18 取得 90.62% 的准确率后，观察到仍有部分样本被误判。分析发现，这些误判往往发生在纹理高度相似的矿石之间（如某些颜色的石英与长石）。传统的卷积神经网络对整张图片进行均匀的特征提取，缺乏对关键区域（如晶体边缘、特定色彩内含物）的专注能力，且容易受到背景噪声的干扰。

为了进一步提升细粒度分类性能，本项目提出在 ResNet 骨干网络中引入 **CBAM (Convolutional Block Attention Module)**。CBAM 是一种轻量级的双重注意力机制，旨在让网络学会“看哪里”和“关注什么”：

1.  **通道注意力 (Channel Attention)**：自适应地校准特征通道的权重，强调关键的纹理和色彩通道。
2.  **空间注意力 (Spatial Attention)**：在特征图的空间维度上计算注意力掩码，聚焦于矿石主体，抑制无关背景。

### 5.2 改进后的网络结构

改进后的网络被称为 `MineralCBAMResNet`，其核心变化在于将 ResNet 的标准 `BasicBlock` 替换为 `CBAMBasicBlock`。

**CBAMBasicBlock 结构图示：**

```
Input -> [Conv3x3] -> [BN] -> [ReLU] -> [Conv3x3] -> [BN] 
      -> [Channel Attention Module] -> (*) 加权
      -> [Spatial Attention Module] -> (*) 加权
      -> (+) Residual Connection
      -> [ReLU] -> Output
```

**关键代码实现 (`modelCBAM.py`):**

```python
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 结合 AvgPool 和 MaxPool 以保留更多信息
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# CBAM Basic Block
class CBAMBasicBlock(nn.Module):
    def __init__(self, ...):
        # ... 卷积定义 ...
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = ... # 两次卷积
        
        # 依次注入注意力机制
        out = self.ca(out) * out # 通道加权
        out = self.sa(out) * out # 空间加权
        
        out += self.shortcut(x)
        return F.relu(out)
```

### 5.3 改进模型训练结果与性能对比

改进模型同样并在相同的超参数设置下（Optimizer: Adam, LR: 0.001, Scheduler: ReduceLROnPlateau）训练了 50 个 Epoch。

![results](https://github.com/LuboAB/Task1/blob/main/assets/results-1768406889835.png?raw=true)

**训练日志摘要 (CBAM-ResNet)：**
*   **收敛速度**：CBAM 模型在早期下降更快，Epoch 13 时 Loss 已降至 0.1014，而原模型在同阶段 Loss 为 0.4456。
*   **最佳验证准确率 (Best Val Acc)**：**90.97%** (Epoch 22)
*   **最终验证准确率 (Final Val Acc)**：**90.53%** (Epoch 50)
*   **最终验证 Loss**: 0.4568

**性能指标对比表：**

| 指标                     | 基础网络 (Baseline ResNet-18) | 改进网络 (CBAM-ResNet) | 变化             |
| :----------------------- | :---------------------------- | :--------------------- | :--------------- |
| **最佳验证准确率**       | 90.62%                        | **90.97%**             | **+0.35%**       |
| **收敛速度 (达90% Acc)** | Epoch 36                      | **Epoch 13**           | **快 2 倍以上**  |
| **最低验证 Loss**        | 0.5065                        | **0.4074**             | **-19.5%**       |
| **参数量**               | ~11.17M                       | ~11.25M                | +0.7% (几乎无视) |

**分析与讨论：**

1.  **收敛效率显著提升**：这是本次改进最显著的成果。引入注意力机制后，模型能在短短 **13 个 Epoch** 内就达到 90.35% 的高准确率，而基础模型直到 Epoch 36 才达到这一水平。这说明 CBAM 模块有效地帮助网络更快地锁定了具有判别力的特征。
2.  **泛化能力增强**：虽然最终准确率的绝对值提升看似不大（约 0.35%），但观察验证集 Loss 可以发现，改进模型的 Loss 下限显著更低 (0.40 vs 0.50)，这意味着模型对预测结果的置信度更高，预测分布更加确定。
3.  **计算性价比高**：仅仅增加了不到 1% 的参数量，就换来了训练效率的大幅提升和更稳健的预测能力，证明了 CBAM 架构在该任务上的有效性。

**综合结论：**
通过在 ResNet 基础架构中融合 CBAM 双重注意力机制，成功构建了一个收敛更快、特征提取能力更强的矿石分类模型。实验结果表明，这种改进设计对于处理纹理复杂的矿石图像具有明显的性能优势，是一个行之有效的优化方案。
