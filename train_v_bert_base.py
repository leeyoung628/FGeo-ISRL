import os
import json
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import warnings

warnings.filterwarnings("ignore")

# ======================= 配置类 =======================
class Config:
    csv_path = "/home/lengmen/liyang/FGeoBDRL/code/value_net/dataset_for_value.csv"
    model_name = "/home/lengmen/liyang/FGeoBDRL/Bert-base"
    model_save_path = "/home/lengmen/liyang/FGeoBDRL/code/value_net/pt/test0.01/value_net.pth"

    batch_size = 16
    max_epochs = 100
    max_length = 256
    lr = 1e-5
    id_emb_dim = 64
    threshold = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= 日志配置 =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ======================= 自定义数据集 =======================
class ValueDataset(Dataset):
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
        reward = torch.tensor(row["reward"], dtype=torch.float32)

        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(qid, dtype=torch.long), reward

# ======================= 模型定义 =======================
class ValueNet(nn.Module):
    def __init__(self, model_name, num_ids, id_emb_dim):
        super(ValueNet, self).__init__()
        self.llm = AutoModel.from_pretrained(model_name)
        llm_hidden = self.llm.config.hidden_size
        self.id_embedding = nn.Embedding(num_ids, id_emb_dim)
        self.regressor = nn.Sequential(
            nn.Linear(llm_hidden + id_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, qid):
        llm_out = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = llm_out.last_hidden_state[:, 0, :]
        id_emb = self.id_embedding(qid)
        combined = torch.cat((pooled_output, id_emb), dim=1)
        return self.regressor(combined).squeeze()

# ======================= Checkpoint 工具 =======================
def manage_checkpoints(ckpt_dir, base_name, max_ckpts=3):
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if base_name in f and f.endswith(".pth") and "_epoch" in f],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )
    while len(ckpt_files) > max_ckpts:
        os.remove(os.path.join(ckpt_dir, ckpt_files.pop(0)))

def find_latest_checkpoint(ckpt_dir, base_name):
    if not os.path.exists(ckpt_dir):
        return None, 0
    ckpt_files = [
        f for f in os.listdir(ckpt_dir)
        if base_name in f and f.endswith(".pth") and "_epoch" in f
    ]
    if not ckpt_files:
        return None, 0
    ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
    latest_ckpt = ckpt_files[-1]
    epoch_num = int(latest_ckpt.split("_epoch")[-1].replace(".pth", ""))
    return os.path.join(ckpt_dir, latest_ckpt), epoch_num

# ======================= 训练 & 验证 =======================
def train_one_epoch(model, dataloader, optimizer, criterion, device, threshold, epoch_num):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(dataloader, desc=f"[Epoch {epoch_num}] Training", ncols=100)
    for input_ids, attention_mask, qid, reward in loop:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        qid, reward = qid.to(device), reward.to(device)

        optimizer.zero_grad()
        pred = model(input_ids, attention_mask, qid)
        loss = criterion(pred, reward)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = ((pred - reward).abs() < threshold).sum().item() / reward.size(0)
        loop.set_postfix(loss=batch_loss, acc=batch_acc)

        total_loss += batch_loss
        correct += ((pred - reward).abs() < threshold).sum().item()
        total += reward.size(0)

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device, threshold):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, qid, reward in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            qid, reward = qid.to(device), reward.to(device)

            pred = model(input_ids, attention_mask, qid)
            loss = criterion(pred, reward)
            total_loss += loss.item()
            correct += ((pred - reward).abs() < threshold).sum().item()
            total += reward.size(0)

    return total_loss / len(dataloader), correct / total

# ======================= 主程序 =======================
def main():
    config = Config()
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

    json_logs = []

    raw_data = pd.read_csv(config.csv_path)
    train_df, val_df = train_test_split(raw_data, test_size=0.3, random_state=42)

    id2idx = {id_: idx for idx, id_ in enumerate(raw_data['id'].unique())}
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader = DataLoader(ValueDataset(train_df, tokenizer, config.max_length, id2idx), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(ValueDataset(val_df, tokenizer, config.max_length, id2idx), batch_size=config.batch_size, shuffle=False)

    model = ValueNet(config.model_name, len(id2idx), config.id_emb_dim).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    ckpt_dir = os.path.dirname(config.model_save_path)
    base_name = os.path.basename(config.model_save_path).replace(".pth", "")
    latest_ckpt_path, start_epoch = find_latest_checkpoint(ckpt_dir, base_name)

    if latest_ckpt_path:
        model.load_state_dict(torch.load(latest_ckpt_path))
        logging.info(f"已从 checkpoint 恢复：{latest_ckpt_path}，继续从 epoch {start_epoch + 1} 训练")
    else:
        start_epoch = 0
        logging.info("未发现 checkpoint，将从头开始训练")

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

    logging.info("开始训练价值网络...")
    for epoch in range(start_epoch, config.max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.device, config.threshold, epoch+1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device, config.threshold)

        print(f"[Epoch {epoch+1}] Summary --> TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        json_logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": current_lr
        })

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{base_name}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            manage_checkpoints(ckpt_dir, base_name, max_ckpts=3)
            logging.info(f"Checkpoint saved at: {ckpt_path}")

    torch.save(model.state_dict(), config.model_save_path)
    logging.info(f"最终模型已保存至：{config.model_save_path}")

    with open(os.path.join(ckpt_dir, "value_train_result.json"), "w") as f:
        json.dump(json_logs, f, indent=4)
    logging.info(f"训练日志已保存至：value_train_result.json")

if __name__ == "__main__":
    main()
