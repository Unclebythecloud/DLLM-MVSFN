import torch
from torch.utils.data import Dataset,DataLoader
import json
import config


class NewsDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 加载 JSON 数据
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # 提取文本字段和标签，并过滤掉任何可能导致问题的数据点
        self.content = []
        self.analyze = []
        self.comments = []
        self.labels = []

        for item in self.data:
            text = item.get('text', '').strip()
            analyze = item.get('analyze', '').strip()
            comments = item.get('comments', [])
            label = item.get('label')

            # 确保内容、分析和评论都不是空的，并且标签是一个有效的整数
            if text and analyze and comments and isinstance(label, int):
                self.content.append(text)
                self.analyze.append(analyze)
                self.comments.append(comments)
                self.labels.append(label)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        content = self.content[idx]
        analyze = self.analyze[idx]
        comments = self.comments[idx]
        label = self.labels[idx]

        # 对于 comments，如果它是字符串列表，我们需要合并成单个字符串或者分别编码每个评论
        if isinstance(comments, list):  # 如果 comments 是列表类型
            # 合并非空评论
            comments = ' '.join([str(c) for c in comments if str(c).strip()])
        
        # 检查是否为空字符串或 None 并替换为默认值
        content = content.strip() or ''
        analyze = analyze.strip() or ''
        comments = comments.strip() or ''

        content_encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        analyze_encoding = self.tokenizer.encode_plus(
            analyze,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        comments_encoding = self.tokenizer.encode_plus(
            comments,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'content_input_ids': content_encoding['input_ids'].flatten(),
            'content_attention_mask': content_encoding['attention_mask'].flatten(),
            'analyze_input_ids': analyze_encoding['input_ids'].flatten(),
            'analyze_attention_mask': analyze_encoding['attention_mask'].flatten(),
            'comments_input_ids': comments_encoding['input_ids'].flatten(),
            'comments_attention_mask': comments_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def get_dataloader(path, tokenizer, mode='train'):
    dataset = NewsDataset(path, tokenizer)
    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    elif mode in ['val','test']:
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Please use 'train', 'val', or 'test'.")
    
    return dataloader