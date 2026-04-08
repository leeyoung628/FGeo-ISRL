import torch
import torch.nn as nn
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict

# ======================= 配置类 =======================
class Config:
    csv_path = "D:\Project\FGeo-ISRL\code\policy_net\dataset_for_policy.csv"
    model_name = "D:\Project\FGeo-ISRL\Bert-base" 
    model_save_path = "D:\Project\FGeo-ISRL\code\policy_net\pt\bert_base_test0.01/model.pth"  # 修改保存路径以区分模型
    batch_size = 16
    max_epochs = 100
    max_length = 256
    lr = 1e-5
    id_emb_dim = 64
    num_classes = 234
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= 日志配置 =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ======================= 自定义数据集 =======================
class PolicyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, id2idx):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id2idx = id2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["state"])
        qid = self.id2idx[row["id"]]
        label = torch.tensor(row["action"], dtype=torch.long)  # for classification

        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(qid, dtype=torch.long), label

# ======================= 模型类 =======================
class PolicyNet(nn.Module):
    def __init__(self, model_name, num_ids, id_emb_dim, num_classes):
        super(PolicyNet, self).__init__()
        self.llm = AutoModel.from_pretrained(model_name)
        hidden = self.llm.config.hidden_size
        self.id_embedding = nn.Embedding(num_ids, id_emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + id_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, qid):
        output = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        id_emb = self.id_embedding(qid)
        combined = torch.cat((output, id_emb), dim=1)
        return self.classifier(combined)  # logits

# ======================= Checkpoint 工具 =======================
def manage_checkpoints(ckpt_dir, base_name, max_ckpts=3):
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if base_name in f and "_epoch" in f],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )
    while len(ckpts) > max_ckpts:
        os.remove(os.path.join(ckpt_dir, ckpts.pop(0)))

def find_latest_checkpoint(ckpt_dir, base_name):
    if not os.path.exists(ckpt_dir):
        return None, 0
    ckpts = [f for f in os.listdir(ckpt_dir) if base_name in f and "_epoch" in f]
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
    latest = ckpts[-1]
    epoch_num = int(latest.split("_epoch")[-1].replace(".pth", ""))
    return os.path.join(ckpt_dir, latest), epoch_num

# ======================= 训练与验证函数 =======================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch_num):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(dataloader, desc=f"[Epoch {epoch_num}] Training", ncols=100)

    for input_ids, attention_mask, qid, labels in loop:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        qid = qid.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, qid)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).sum().item() / labels.size(0)

        loop.set_postfix(OrderedDict([
            ("loss", f"{loss.item():<8.4f}"),
            ("acc", f"{batch_acc:<8.4f}")
        ]))

        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, qid, labels in dataloader:
            input_ids, attention_mask, qid, labels = input_ids.to(device), attention_mask.to(device), qid.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, qid)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), correct / total

# ======================= 主程序 =======================
def main():
    config = Config()
    raw_data = pd.read_csv(config.csv_path)
    train_df, val_df = train_test_split(raw_data, test_size=0.3, random_state=42)

    id2idx = {id_: idx for idx, id_ in enumerate(raw_data['id'].unique())}
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader = DataLoader(PolicyDataset(train_df, tokenizer, config.max_length, id2idx), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(PolicyDataset(val_df, tokenizer, config.max_length, id2idx), batch_size=config.batch_size, shuffle=False)

    model = PolicyNet(config.model_name, len(id2idx), config.id_emb_dim, config.num_classes).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    ckpt_dir = os.path.dirname(config.model_save_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    base_name = os.path.basename(config.model_save_path).replace(".pth", "")
    latest_ckpt, start_epoch = find_latest_checkpoint(ckpt_dir, base_name)

    if latest_ckpt:
        model.load_state_dict(torch.load(latest_ckpt))
        logging.info(f"恢复训练：{latest_ckpt}")
    else:
        start_epoch = 0
        logging.info("未发现 checkpoint，将从头开始训练")

    train_loss_list, val_loss_list, train_acc_list, val_acc_list, logs = [], [], [], [], []

    for epoch in range(start_epoch, config.max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.device, epoch+1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)

        print(f"[Epoch {epoch+1}] Summary --> TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{base_name}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            manage_checkpoints(ckpt_dir, base_name)
            logging.info(f"保存 checkpoint: {ckpt_path}")

    torch.save(model.state_dict(), config.model_save_path)
    logging.info(f"训练完成，保存最终模型到 {config.model_save_path}")

    with open(os.path.join(ckpt_dir, "policy_train_result.json"), "w") as f:
        json.dump(logs, f, indent=4)

if __name__ == "__main__":
    main()
