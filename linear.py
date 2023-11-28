import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import seaborn as sns


# Resample & reorg train—val set
import matplotlib.pyplot as plt

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

input_size = 23  # 输入特征的数量
hidden_size = 1024  # 隐藏层大小
output_size = 14  # 输出类别的数量

dataarray = np.load('/Users/cszy98/PycharmProjects/diffusion_for_seg/dataarray.npy')
dataarray = np.array(dataarray,dtype=np.float32)
dataarray = np.random.permutation(dataarray)
from sklearn.model_selection import train_test_split

# X 是特征，y 是标签
# X_train, X_test, y_train, y_test = train_test_split(dataarray[:,:23], dataarray[:,23], test_size=0.2, stratify=dataarray[:,23])
# # print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# y_train = dataarray[:,23]

# statistics = {}
# for i in range(y_train.shape[0]):
#     if y_train[i] not in statistics:
#         statistics[y_train[i]] = [i]
#     else:
#         statistics[y_train[i]].append(i)
# for k in statistics.keys():
#     print(k,len(statistics[k]))
# print('&&&&&&&&&&&&&&&&&&')
# statistics = {}
# for i in range(y_test.shape[0]):
#     if y_test[i] not in statistics:
#         statistics[y_test[i]] = [i]
#     else:
#         statistics[y_test[i]].append(i)
# for k in statistics.keys():
#     print(k,len(statistics[k]))
# exit(0)

# dataarray = np.random.permutation(dataarray)

# print(dataarray.shape)
# data = torch.tensor(dataarray[:,:23])#torch.randn(N, input_size)
# labels = torch.tensor(dataarray[:,-1]).reshape((1357,)).long() #torch.randint(0, output_size, (N,))
# print(labels.shape)
# N = data.shape[0]

# # 划分数据集为训练集和测试集
# train_size = int(0.8 * N)


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = dataarray[:,:23]
y = dataarray[:,23]
foldn = 0
for train_index, test_index in skf.split(X, y):
    foldn+=1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # statistics = {}
    # for i in range(y_train.shape[0]):
    #     if y_train[i] not in statistics:
    #         statistics[y_train[i]] = [i]
    #     else:
    #         statistics[y_train[i]].append(i)
    # for k in statistics.keys():
    #     print(k,len(statistics[k]))
    # print('&&&&&&&&&&&&&&&&&&')
    # statistics = {}
    # for i in range(y_test.shape[0]):
    #     if y_test[i] not in statistics:
    #         statistics[y_test[i]] = [i]
    #     else:
    #         statistics[y_test[i]].append(i)
    # for k in statistics.keys():
    #     print(k,len(statistics[k]))
    # print('&&&&&&&&&&&&&&&&&&***************')
    # continue

    train_data, test_data = torch.tensor(X_train).float(),torch.tensor(X_test).float()
    train_labels, test_labels = torch.tensor(y_train).long(), torch.tensor(y_test).long()
    # 转换为 PyTorch 的 TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # 使用 DataLoader 加载数据
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    num_epochs =500
    # 初始化模型、损失函数和优化器
    model = MLPClassifier(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    train_losses = []
    val_losses = []
    acc = []
    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 10 ==0: # or epoch==499:
            train_losses.append(loss.item())
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                loss_val = 0
                val_cnt = 0
                predlist = []
                realist = []
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss_val += criterion(outputs, labels)
                    val_cnt+=1
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    predlist.append(predicted)
                    realist.append(labels)
                from sklearn.metrics import confusion_matrix
                conf_matrix = confusion_matrix(torch.cat(realist).numpy(), torch.cat(predlist).numpy())
                print(conf_matrix.shape)

                accuracy = correct / total
                acc.append(accuracy)
                val_losses.append(loss_val/val_cnt)
                print(f'Test Accuracy: {accuracy * 100:.2f}%')
                print('val loss', loss_val/val_cnt)
            model.train()
    print(conf_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    plt.close()
    
    torch.save(model.state_dict(),'CodingTest/view/'+str(foldn)+'.pth')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch*10')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('CodingTest/view/'+str(foldn)+'_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(acc, label='Validation Accuracy')
    plt.xlabel('Epoch*10')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('CodingTest/view/'+str(foldn)+'_accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
