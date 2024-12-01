import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据加载与预处理
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)  # 无列名
    labels = df.iloc[:, 0].values  # 第一列是标签
    # 假设第二列和第三列是文本内容，合并它们作为输入
    texts = df.iloc[:, 1] + ' ' + df.iloc[:, 2]  # 第二列和第三列是文本
    return texts, labels


def tokenize(text):
    return text.split()  # 使用空格分割文本，如果需要更复杂的分词方法可以替换为其他方法


def build_vocab(texts):
    vocab = set(word for text in texts for word in tokenize(text))
    return {word: idx + 1 for idx, word in enumerate(vocab)}  # 从1开始编码，0留给填充符


def text_to_sequence(text, vocab):
    return [vocab.get(word, 0) for word in tokenize(text)]  # 将文本转换为词汇表索引


# 自定义 Dataset 和 collate_fn
class NewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)

    # 计算每个文本的长度
    lengths = [len(seq) for seq in sequences]

    # 填充序列，使其具有相同长度
    padded_sequences = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in sequences],
                                    batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels, lengths


# 模型定义
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.rnn(packed_embedded)

        # 如果是双向LSTM，将正反向隐藏状态拼接
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(hidden)


# 数据加载
train_texts, train_labels = load_data('train.csv')
test_texts, test_labels = load_data('test.csv')

# 构建词汇表并转换数据
vocab = build_vocab(train_texts)
train_sequences = [text_to_sequence(text, vocab) for text in train_texts]
test_sequences = [text_to_sequence(text, vocab) for text in test_texts]
train_labels = [label - 1 for label in train_labels]  # 假设标签从1开始
test_labels = [label - 1 for label in test_labels]

# 划分训练集和验证集
train_sequences, val_sequences, train_labels, val_labels = train_test_split(train_sequences, train_labels,
                                                                            test_size=0.2, random_state=42)

# 构建 DataLoader
train_loader = DataLoader(NewsDataset(train_sequences, train_labels), batch_size=32, shuffle=True,
                          collate_fn=collate_fn)
val_loader = DataLoader(NewsDataset(val_sequences, val_labels), batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(NewsDataset(test_sequences, test_labels), batch_size=32, shuffle=False, collate_fn=collate_fn)

# 初始化模型
vocab_size = len(vocab) + 1  # vocab索引从1开始，因此大小需要+1
embedding_dim = 100
hidden_dim = 128
output_dim = 4  # 假设有4个类别
n_layers = 2
bidirectional = True
dropout = 0.5

model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout).to(
    device)  # 移动模型到GPU
optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# 训练与验证
def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for text, labels, lengths in loader:
        text, labels, lengths = text.to(device), labels.to(device), torch.tensor(lengths).to(device)

        # lengths 需要移到 CPU，但其他数据保留在 GPU
        lengths = lengths.cpu()  # 修复这里

        optimizer.zero_grad()
        predictions = model(text, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for text, labels, lengths in loader:
            text, labels, lengths = text.to(device), labels.to(device), torch.tensor(lengths).to(device)

            # lengths 需要移到 CPU，但其他数据保留在 GPU
            lengths = lengths.cpu()  # 修复这里

            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


# 训练模型
n_epochs = 10
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# 测试模型
model.eval()
all_preds, all_labels = [], []
all_probs = []  # 存储每个类别的概率预测值，用于ROC和AUC的计算

with torch.no_grad():
    for text, labels, lengths in test_loader:
        text, labels, lengths = text.to(device), labels.to(device), torch.tensor(lengths).to(device)
        lengths = lengths.cpu()  # 修复这里

        predictions = model(text, lengths)
        preds = torch.argmax(predictions, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 获取每个类别的概率预测值
        probabilities = torch.softmax(predictions, dim=1)
        all_probs.extend(probabilities.cpu().numpy())

# 性能指标与混淆矩阵
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
conf_matrix = confusion_matrix(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')

# 计算ROC和AUC
# all_labels需要是二进制形式，如果是多分类问题，需要转换为二进制
# 这里假设all_labels和all_preds已经是二进制形式或者您已经进行了转换

print(f"Test Accuracy: {accuracy:.6f}, F1 Score: {f1:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# 绘制损失曲线
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.savefig('/home/shisr/Data/hw/machinehw11/train_2.jpg', format='jpg')

n_classes = 4  # 假设有4个类别
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels, [prob[:, i] for prob in all_probs])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制所有类别的ROC曲线
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class')
plt.legend(loc="lower right")
plt.show()

# 保存ROC曲线图像
plt.savefig('/home/shisr/Data/hw/machinehw11/roc_curve.jpg', format='jpg')