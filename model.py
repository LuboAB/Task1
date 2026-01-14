import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torchvision

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
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

def MineralCNN(num_classes=7):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def get_next_run_dir(base_dir='runs'):
    base_path = Path(base_dir)
    if not base_path.exists():
        new_dir = base_path / 'train'
    else:
        # 查找现有的 train* 文件夹
        existing_dirs = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith('train')]
        if not existing_dirs:
            new_dir = base_path / 'train'
        else:
            # 提取所有后缀数字
            nums = []
            for d in existing_dirs:
                suffix = d.replace('train', '')
                if suffix == '':
                    nums.append(1)
                elif suffix.isdigit():
                    nums.append(int(suffix))
            
            next_num = max(nums) + 1 if nums else 1
            new_dir = base_path / f'train{next_num}'
            
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir

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