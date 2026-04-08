import torch
import torch.nn as nn
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict

# 导入必要的函数（假设这些函数已经存在于相应的模块中）
import sys
sys.path.append("/home/lengmen/liyang/FGeoBDRL/code/policy_net")
sys.path.append("/home/lengmen/liyang/FGeoBDRL/code/value_net")
sys.path.append("D:\Project\FGeo-ISRL\code\mcts_net")
from mcts_net.add_premise import add_premise
from mcts_net.JudgeSolveOrNot import solve_or_not
from mcts_net.load_param_by_problem_id_and_theorem import load_param_by_problem_id_and_theorem
from mcts_net.load_final_state import load_final_state

# ======================= 配置类 =======================
class Config:
    csv_path = "/home/lengmen/liyang/FGeoBDRL/code/policy_net/dataset_for_policy.csv"
    model_name = "/home/lengmen/liyang/FGeoBDRL/DistillBert"
    model_save_path = "/home/lengmen/liyang/FGeoBDRL/code/policy_net/pt/test_mcts/model.pth"
    batch_size = 16
    max_epochs = 15
    max_length = 256
    lr = 2e-5
    id_emb_dim = 64
    num_classes = 235
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 添加num_ids属性，用于哈希ID
    num_ids = 10000  # 设置一个足够大的值用于哈希
    
    # MCTS相关配置
    use_mcts = True
    mcts_simulations = 10  # 每次MCTS搜索的模拟次数
    mcts_c_puct = 1.5     # MCTS的探索常数
    mcts_temperature = 1.0  # 温度参数，控制探索与利用的平衡
    mcts_dirichlet_noise = 0.3  # Dirichlet噪声参数
    mcts_noise_alpha = 0.03     # Dirichlet噪声的alpha值
    mcts_update_interval = 5    # 每隔多少个batch更新一次MCTS策略

# ======================= 日志配置 =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ======================= MCTS节点类 =======================
class MCTSNode:
    """蒙特卡洛树搜索的节点类"""
    def __init__(self, state, parent=None, action=None, param=None, prior_p=0.0):
        self.state = state  # 几何状态（文本形式）
        self.parent = parent  # 父节点
        self.action = action  # 到达此节点的动作（定理编号）
        self.param = param    # 应用动作的参数
        self.children = []    # 子节点列表
        self.visits = 0       # 访问次数
        self.value_sum = 0.0  # 累积价值
        self.prior_p = prior_p  # 先验概率（来自策略网络）
        
    def expand(self, actions, params, priors):
        """扩展当前节点，添加子节点"""
        for action, param, prior in zip(actions, params, priors):
            # 创建新的子节点
            child = MCTSNode(
                state=self.state,  # 初始状态与父节点相同，会在模拟时更新
                parent=self,
                action=action,
                param=param,
                prior_p=prior
            )
            self.children.append(child)
        
    def select(self, c_puct=1.0):
        """使用UCB公式选择最有价值的子节点"""
        # 选择UCB值最高的子节点
        return max(self.children, key=lambda n: n.get_ucb(c_puct))
    
    def update(self, value):
        """更新节点统计信息"""
        self.visits += 1
        self.value_sum += value
        
    def get_value(self):
        """获取节点的平均价值"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def get_ucb(self, c_puct):
        """计算UCB值"""
        if self.visits == 0:
            return float('inf')  # 未访问过的节点优先级最高
        
        # UCB公式：Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploitation = self.get_value()
        exploration = c_puct * self.prior_p * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration
    
    def is_leaf(self):
        """判断是否为叶节点"""
        return len(self.children) == 0
    
    def is_root(self):
        """判断是否为根节点"""
        return self.parent is None

# ======================= MCTS类 =======================
class MCTS:
    """蒙特卡洛树搜索类"""
    def __init__(self, policy_model, policy_tokenizer, config, add_premise_func, solve_or_not_func, load_param_func):
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.config = config
        self.add_premise = add_premise_func
        self.solve_or_not = solve_or_not_func
        self.load_param = load_param_func
        self.c_puct = config.mcts_c_puct
        self.temperature = config.mcts_temperature
        self.dirichlet_noise = config.mcts_dirichlet_noise
        self.noise_alpha = config.mcts_noise_alpha
        
    def predict_policy(self, state, problem_id, top_k=10):
        """使用策略网络预测动作概率"""
        state_for_prediction = state.strip()
        id_for_prediction = str(problem_id)
        
        # 获取策略网络预测的动作
        logits = self._get_policy_logits(state_for_prediction, id_for_prediction)
        probs = torch.softmax(logits, dim=1)[0].cpu().detach().numpy()
        
        # 获取前k个最高概率的动作
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        
        return top_indices, top_probs
    
    def _get_policy_logits(self, state_text, id_text):
        """获取策略网络的logits输出"""
        self.policy_model.eval()
        with torch.no_grad():
            qid_idx = hash(id_text) % self.config.num_ids
            qid_tensor = torch.tensor([qid_idx], dtype=torch.long).to(self.config.device)
            encoded = self.policy_tokenizer(
                state_text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.config.device)
            attention_mask = encoded["attention_mask"].to(self.config.device)
            return self.policy_model(input_ids, attention_mask, qid_tensor)
    
    def search(self, root_state, problem_id, final_state, n_simulations=50):
        """执行MCTS搜索"""
        # 创建根节点
        root = MCTSNode(state=root_state)
        
        # 如果初始状态已经解决问题，直接返回
        if self.solve_or_not(final_state, root_state):
            return None, None, np.zeros(self.config.num_classes)
        
        # 获取可能的动作和参数
        actions, action_probs = self.predict_policy(root_state, problem_id)
        
        # 添加Dirichlet噪声以增加探索
        if self.dirichlet_noise > 0:
            noise = np.random.dirichlet([self.noise_alpha] * len(actions))
            action_probs = (1 - self.dirichlet_noise) * action_probs + self.dirichlet_noise * noise
        
        # 筛选有效的动作和参数
        valid_actions = []
        valid_params = []
        valid_probs = []
        
        for action, prob in zip(actions, action_probs):
            param_list = self.load_param(problem_id, action)
            if param_list:
                if isinstance(param_list, list):
                    param = param_list[0]
                else:
                    param = param_list
                
                # 检查是否可以应用
                try:
                    new_premise = self.add_premise(action, param)
                    if new_premise:
                        valid_actions.append(action)
                        valid_params.append(param)
                        valid_probs.append(prob)
                except:
                    continue
        
        # 如果没有有效动作，返回空结果
        if not valid_actions:
            return None, None, np.zeros(self.config.num_classes)
        
        # 归一化概率
        if valid_probs:
            valid_probs = np.array(valid_probs)
            valid_probs = valid_probs / np.sum(valid_probs)
        
        # 扩展根节点
        root.expand(valid_actions, valid_params, valid_probs)
        
        # 执行多次模拟
        for _ in range(n_simulations):
            # 选择阶段：从根节点选择到叶节点
            node = root
            search_path = [node]
            
            # 选择阶段
            while not node.is_leaf():
                node = node.select(self.c_puct)
                search_path.append(node)
            
            # 如果叶节点已经被访问过，则扩展
            if node.visits > 0 and not node.is_root():
                # 应用动作获取新状态
                try:
                    new_premise = self.add_premise(node.action, node.param)
                    if new_premise:
                        new_state = new_premise + "," + node.parent.state
                        node.state = new_state
                        
                        # 检查是否解决问题
                        if self.solve_or_not(final_state, new_state):
                            # 如果解决了问题，给予高奖励
                            value = 1.0
                            # 回溯更新
                            for n in reversed(search_path):
                                n.update(value)
                            continue
                        
                        # 获取新状态的动作概率
                        actions, action_probs = self.predict_policy(new_state, problem_id)
                        
                        # 筛选有效的动作和参数
                        valid_actions = []
                        valid_params = []
                        valid_probs = []
                        
                        for action, prob in zip(actions, action_probs):
                            param_list = self.load_param(problem_id, action)
                            if param_list:
                                if isinstance(param_list, list):
                                    param = param_list[0]
                                else:
                                    param = param_list
                                
                                # 检查是否可以应用
                                try:
                                    new_premise = self.add_premise(action, param)
                                    if new_premise:
                                        valid_actions.append(action)
                                        valid_params.append(param)
                                        valid_probs.append(prob)
                                except:
                                    continue
                        
                        # 如果有有效动作，扩展节点
                        if valid_actions:
                            # 归一化概率
                            valid_probs = np.array(valid_probs)
                            valid_probs = valid_probs / np.sum(valid_probs)
                            
                            # 扩展节点
                            node.expand(valid_actions, valid_params, valid_probs)
                    else:
                        # 如果无法应用动作，给予低奖励
                        value = 0.0
                        # 回溯更新
                        for n in reversed(search_path):
                            n.update(value)
                        continue
                except:
                    # 出错时给予低奖励
                    value = 0.0
                    # 回溯更新
                    for n in reversed(search_path):
                        n.update(value)
                    continue
            
            # 模拟阶段：随机模拟到游戏结束
            # 在这个问题中，我们使用简单的启发式方法：
            # 如果当前状态已经解决问题，给予高奖励；否则给予中等奖励
            if node.is_leaf() and not node.is_root():
                try:
                    new_premise = self.add_premise(node.action, node.param)
                    if new_premise:
                        new_state = new_premise + "," + node.parent.state
                        node.state = new_state
                        
                        # 检查是否解决问题
                        if self.solve_or_not(final_state, new_state):
                            value = 1.0
                        else:
                            # 使用简单启发式：随机值
                            value = 0.5
                    else:
                        value = 0.0
                except:
                    value = 0.0
            else:
                # 如果节点已经被扩展，使用随机值
                value = 0.5
            
            # 回溯阶段：更新搜索路径上的所有节点
            for n in reversed(search_path):
                n.update(value)
        
        # 根据访问次数计算动作概率
        action_probs = np.zeros(self.config.num_classes)
        for child in root.children:
            # 使用温度参数调整访问次数
            action_probs[child.action] = child.visits ** (1 / self.temperature)
        
        # 归一化概率
        if np.sum(action_probs) > 0:
            action_probs = action_probs / np.sum(action_probs)
        
        # 选择最佳动作
        if root.children:
            best_child = max(root.children, key=lambda n: n.visits)
            best_action = best_child.action
            best_param = best_child.param
        else:
            best_action = None
            best_param = None
        
        return best_action, best_param, action_probs

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
        return input_ids, attention_mask, torch.tensor(qid, dtype=torch.long), label, row["id"], text

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
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch_num, config, mcts=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    # 减少进度条输出信息，只显示epoch和进度
    loop = tqdm(dataloader, desc=f"Epoch {epoch_num}", ncols=80, leave=False)

    for batch_idx, (input_ids, attention_mask, qid, labels, problem_ids, states) in enumerate(loop):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        qid = qid.to(device)
        labels = labels.to(device)

        # 是否使用MCTS生成策略标签
        if config.use_mcts and mcts and batch_idx % config.mcts_update_interval == 0:
            # 使用MCTS生成策略标签
            mcts_labels = []
            for i in range(len(problem_ids)):
                problem_id = problem_ids[i].item()
                state = states[i]
                
                # 加载最终状态
                final_state = load_final_state(problem_id)
                
                if final_state:
                    # 执行MCTS搜索
                    _, _, action_probs = mcts.search(
                        root_state=state,
                        problem_id=problem_id,
                        final_state=final_state,
                        n_simulations=config.mcts_simulations
                    )
                    
                    # 将MCTS生成的策略转换为标签
                    mcts_label = torch.tensor(action_probs, dtype=torch.float32).to(device)
                else:
                    # 如果没有最终状态，使用原始标签
                    mcts_label = torch.zeros(config.num_classes, dtype=torch.float32).to(device)
                    mcts_label[labels[i]] = 1.0
                
                mcts_labels.append(mcts_label)
            
            # 将MCTS标签堆叠成批次
            mcts_labels = torch.stack(mcts_labels)
            
            # 使用KL散度损失
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, qid)
            probs = torch.softmax(logits, dim=1)
            
            # 计算KL散度损失
            log_probs = torch.log(probs + 1e-10)
            loss = -torch.sum(mcts_labels * log_probs) / len(mcts_labels)
        else:
            # 使用标准交叉熵损失
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, qid)
            loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        
        # 更新统计信息但不在进度条中显示
        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 简化进度条显示的信息
        loop.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, qid, labels, _, _ in dataloader:
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
    # 更新config中的num_ids为实际的唯一ID数量
    config.num_ids = len(id2idx)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader = DataLoader(
        PolicyDataset(train_df, tokenizer, config.max_length, id2idx),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        PolicyDataset(val_df, tokenizer, config.max_length, id2idx),
        batch_size=config.batch_size,
        shuffle=False
    )

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

    # 初始化MCTS
    if config.use_mcts:
        mcts = MCTS(
            policy_model=model,
            policy_tokenizer=tokenizer,
            config=config,
            add_premise_func=add_premise,
            solve_or_not_func=solve_or_not,
            load_param_func=load_param_by_problem_id_and_theorem
        )
        logging.info("已初始化MCTS用于训练")
    else:
        mcts = None

    train_loss_list, val_loss_list, train_acc_list, val_acc_list, logs = [], [], [], [], []
    
    # 定义JSON文件路径
    json_path = os.path.join(ckpt_dir, "policy_train_result.json")
    csv_path = os.path.join(ckpt_dir, "training_metrics.csv")
    
    # 如果恢复训练，尝试加载已有的训练日志
    if start_epoch > 0 and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_logs = json.load(f)
                # 提取已有的训练记录
                if "detailed_logs" in existing_logs:
                    logs = existing_logs["detailed_logs"]
                    # 确保只保留已完成epoch的记录
                    logs = [log for log in logs if log["epoch"] <= start_epoch]
                    # 从logs中提取训练指标
                    train_loss_list = [log["train_loss"] for log in logs]
                    val_loss_list = [log["val_loss"] for log in logs]
                    train_acc_list = [log["train_acc"] for log in logs]
                    val_acc_list = [log["val_acc"] for log in logs]
                    logging.info(f"已加载{len(logs)}个epoch的训练记录")
        except Exception as e:
            logging.warning(f"加载训练日志失败: {e}")
            # 如果加载失败，重新初始化
            train_loss_list, val_loss_list, train_acc_list, val_acc_list, logs = [], [], [], [], []

    for epoch in range(start_epoch, config.max_epochs):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.device, epoch+1, config, mcts)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)

        # 只在每个epoch结束后打印一次摘要信息
        logging.info(f"Epoch {epoch+1}/{config.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        # 记录当前epoch的训练信息
        current_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr'],
            "use_mcts": config.use_mcts
        }
        logs.append(current_log)

        scheduler.step(val_loss)

        # 保存checkpoint但不打印详细信息
        ckpt_path = os.path.join(ckpt_dir, f"{base_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        
        # 每完成一个epoch就更新JSON文件
        final_logs = {
            "epochs": list(range(1, epoch+2)),  # 从1到当前epoch
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "train_acc": train_acc_list,
            "val_acc": val_acc_list,
            "learning_rates": [log["lr"] for log in logs],
            "use_mcts": config.use_mcts,
            "detailed_logs": logs,  # 保留原始的详细日志
            "last_updated": epoch + 1  # 记录最后更新的epoch
        }
        
        # 保存JSON文件
        with open(json_path, "w") as f:
            json.dump(final_logs, f, indent=4)
        
        # 更新CSV文件 - 修复数组长度不一致的问题
        csv_data = pd.DataFrame({
            "epoch": range(1, len(train_loss_list) + 1),
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "train_acc": train_acc_list,
            "val_acc": val_acc_list,
            "learning_rate": [log["lr"] for log in logs]
        })
        csv_data.to_csv(csv_path, index=False)
        
        # 每5个epoch打印一次checkpoint和日志保存信息
        if (epoch + 1) % 5 == 0:
            logging.info(f"已保存checkpoint: {ckpt_path}")
            logging.info(f"已更新训练日志: {json_path}")

    # 保存最终模型
    torch.save(model.state_dict(), config.model_save_path)
    logging.info(f"训练完成，最终模型已保存到 {config.model_save_path}")
    
    # 绘制并保存loss和acc曲线
    plt.figure()
    plt.plot(range(1, config.max_epochs+1), train_loss_list, label='Train Loss')
    plt.plot(range(1, config.max_epochs+1), val_loss_list, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ckpt_dir, "loss_curve.png"))

    plt.figure()
    plt.plot(range(1, config.max_epochs+1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, config.max_epochs+1), val_acc_list, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ckpt_dir, "accuracy_curve.png"))
    
    logging.info("训练日志和图表已保存")


if __name__ == "__main__":
    main()