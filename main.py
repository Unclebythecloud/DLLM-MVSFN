from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import torch, os
import torch.nn as nn
import pandas as pd
from transformers import BertModel, BertTokenizer
from model import SLMModel
from data_loader import get_dataloader
import config

# 随机种子
def initseed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

initseed()

# 定义训练函数
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    # 使用 tqdm 包装数据加载器，显示训练进度
    for batch in tqdm(data_loader, desc="Training", leave=False):
        content_input_ids = batch['content_input_ids'].to(device)
        content_attention_mask = batch['content_attention_mask'].to(device)
        analyze_input_ids = batch['analyze_input_ids'].to(device)
        analyze_attention_mask = batch['analyze_attention_mask'].to(device)
        comments_input_ids = batch['comments_input_ids'].to(device)
        comments_attention_mask = batch['comments_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 前向传播
        logits = model(content_input_ids, content_attention_mask, analyze_input_ids, analyze_attention_mask, comments_input_ids, comments_attention_mask)

        # 计算损失
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


# 定义评估函数
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            content_input_ids = batch['content_input_ids'].to(device)
            content_attention_mask = batch['content_attention_mask'].to(device)
            analyze_input_ids = batch['analyze_input_ids'].to(device)
            analyze_attention_mask = batch['analyze_attention_mask'].to(device)
            comments_input_ids = batch['comments_input_ids'].to(device)
            comments_attention_mask = batch['comments_attention_mask'].to(device)

            labels = batch['labels'].to(device)

            # 前向传播
            logits = model(content_input_ids, content_attention_mask, analyze_input_ids, analyze_attention_mask, comments_input_ids, comments_attention_mask)
            
            # 预测类别
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 预测概率
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # 获取属于正类的概率
            all_probs.extend(probs)

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 计算 F1 分数（宏平均 F1 和针对真实类/虚假类的 F1）
    f1_real = f1_score(all_labels, all_preds, pos_label=0)  # 真实新闻 (pos_label=0)
    f1_fak = f1_score(all_labels, all_preds, pos_label=1)  # 虚假新闻 (pos_label=1)
    macF1 = f1_score(all_labels, all_preds, average='macro')  # 宏平均 F1

    # 计算 AUC（对正类概率计算 AUC）
    auc = roc_auc_score(all_labels, all_probs)

    return accuracy, macF1, auc, f1_real, f1_fak


def main():    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径
    train_json_path = config.TRAIN_JSON_PATH
    dev_json_path = config.DEV_JSON_PATH
    test_json_path = config.TEST_JSON_PATH
    bert_path = config.BERT_PATH

    # 加载BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    train_loader = get_dataloader(train_json_path, tokenizer, mode='train')
    dev_loader = get_dataloader(dev_json_path, tokenizer, mode='val')
    test_loader = get_dataloader(test_json_path, tokenizer, mode='test')

    # 加载BERT特征提取器
    bert_model = BertModel.from_pretrained(bert_path)  # 不使用分类头
    model = SLMModel(bert_model).to(device)

    # 定义优化器和损失函数
    # optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # 使用加权交叉熵损失函数
    # weights = torch.tensor([1.0, 4.0]).to(device)  # 可以根据需要调整权重
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()


    # 训练模型
    for epoch in range(config.EPOCHS):  
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss}")
        # 每个epoch保存模型
        torch.save(model.state_dict(), f"{os.path.join(config.MODEL_SAVE_PATH, 'bertmodel_epoch_{epoch + 1}.pt')}")


    # 加载每个模型在测试集上进行最终评估
    for epoch in range(config.EPOCHS): 
        model.load_state_dict(torch.load(f"{os.path.join(config.MODEL_SAVE_PATH, 'bertmodel_epoch_{epoch+1}.pt')}"))
        test_accuracy, test_macF1, test_auc, test_f1_real, test_f1_fak = evaluate(model, test_loader, device)
        # test_accuracy, test_macF1, test_auc, test_f1_real, test_f1_fak = evaluate(model, test_loader, device)

        # 打印一个表格展示所有评估结果
        result_df = pd.DataFrame({
            'Metric': ['Accuracy', 'macF1', 'AUC', 'F1-real', 'F1-fak'],
            'Test Set': [test_accuracy, test_macF1, test_auc, test_f1_real, test_f1_fak]
        })

        print("\nEvaluation Results:")
        print(result_df)

if __name__ == '__main__':
    main()