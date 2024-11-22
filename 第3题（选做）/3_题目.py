import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np




class ExperimentBase:
    def __init__(self, device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.prepare_data()
        
    def prepare_data(self):
        
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class NormalizationExperiment(ExperimentBase):
    def __init__(self, device):
        super().__init__(device)
        self.results = {}
        
    def create_model(self, norm_type='none'):
        """
        实验题目1：完成不同归一化方式的模型构建
        
        要求：
        1. 实现带有不同归一化方式的神经网络模型
        2. 支持的归一化方式包括：BatchNorm (bn), LayerNorm (ln), InstanceNorm (in)
        3. 模型结构：784 -> 128 -> 64 -> 10
        """
        class NeuralNetwork(nn.Module):
            def __init__(self, norm_type):
                super(NeuralNetwork, self).__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(_*_, 128)    #补充参数
                self.relu1 = nn.ReLU()
                
                # 使用if—else结构添加不同类型的归一化层
                










                    
                self.fc2 = nn.Linear(128, 64)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(64, 10)
                
            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.norm1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                x = self.norm2(x)
                x = self.relu2(x)
                x = self.fc3(x)
                return x
                
        return NeuralNetwork(norm_type)
    
    def train_model(self, model, epochs=5):
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_loader):.4f}")
    
    def evaluate_model(self, model):
  
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"测试集准确率: {accuracy:.4f}")
        return accuracy
    
    def visualize_features(self, model, norm_type, num_samples=2000):
        """
        实验题目2：实现t—SNE特征降维
        
        要求：
        1. 使用t-SNE进行特征降维
        2. 绘制降维后的特征分布图
        3. 使用不同颜色标识不同类别
        """
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in self.train_loader:
                images = images.to(self.device)
                x = model.flatten(images)
                x = model.relu1(model.norm1(model.fc1(x)))
                x = model.relu2(model.norm2(model.fc2(x)))
                features.append(x.cpu().numpy())
                labels.append(label.numpy())
                if len(features) * self.train_loader.batch_size >= num_samples:
                    break

        features = np.concatenate(features, axis=0)[:num_samples]
        labels = np.concatenate(labels, axis=0)[:num_samples]

                                                                  #tsne设置

        features_embedded = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=features_embedded[:, 0], y=features_embedded[:, 1], 
                       hue=labels, palette="tab10", legend="full", alpha=0.8)
        plt.title(f"t-SNE 可视化 ({norm_type} 归一化)")
        plt.xlabel("t-SNE 维度 1")
        plt.ylabel("t-SNE 维度 2")
        plt.legend(title="类别", loc="best")
        plt.show()

    def run_experiment(self):
        """
        实验题目3：对比不同归一化
    
        要求：
        1. 对比不同归一化方式的性能
            2. 记录并展示实验结果
        3. 分析不同归一化方式的优劣
        """
        norm_types =         #分析不同归一化方式
        epochs_loss = {norm_type: [] for norm_type in norm_types}
    
        for norm_type in norm_types:
            print(f"\n开始 {norm_type} 归一化实验...")
        
            
            model = self.create_model(norm_type).to(self.device)
        
            
            losses = []
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(5):
                model.train()
                running_loss = 0.0
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                epoch_loss = running_loss/len(self.train_loader)
                epochs_loss[norm_type].append(epoch_loss)
                print(f"Epoch {epoch+1}/5, Loss: {epoch_loss:.4f}")
            
            
            accuracy = self.evaluate_model(model)
            self.results[norm_type] = accuracy
            
            
            self.visualize_features(model, norm_type)

        
        plt.figure(figsize=(10, 6))
        for norm_type in norm_types:
            plt.plot(range(1, 6), epochs_loss[norm_type], marker='o', label=norm_type)
        plt.title('不同归一化方式的训练损失对比')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
                                     #绘制loss
        plt.show()

        
        plt.figure(figsize=(10, 6))
        accuracies =                       #从 self.results 字典中提取与 norm_types 列表中每个规范类型（norm_type）相对应的值
        plt.bar(norm_types, accuracies)
        plt.title('不同归一化方式的测试准确率对比')
        plt.xlabel('归一化方式')
        plt.ylabel('准确率')
        for i, acc in enumerate(accuracies):
            plt.text(i, acc, f'{acc:.4f}', ha='center', va='bottom')
        plt.show()

        
        print("\n=== 实验结果汇总 ===")
        print("\n1. 最终准确率对比：")
        for norm_type, accuracy in self.results.items():
            print(f"{norm_type} 归一化准确率: {accuracy:.4f}")
        
        print("\n2. 最终损失值对比：")
        for norm_type in norm_types:
            print(f"{norm_type} 归一化最终损失: {epochs_loss[norm_type][-1]:.4f}")
        
        print("\n3. 性能分析：")
        best_acc = max(self.results.values())
        best_method = [k for k, v in self.results.items() if v == best_acc][0]
        print(f"最佳性能方法: {best_method} (准确率: {best_acc:.4f})")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = NormalizationExperiment(device)
    experiment.run_experiment()