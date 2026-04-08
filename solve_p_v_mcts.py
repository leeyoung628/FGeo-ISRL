import sys
import json
import torch
import torch.nn as nn
import numpy as np
import math
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

# 添加策略网络和价值网络所在目录到系统路径

sys.path.append("D:\Project\FGeo-ISRL\code\policy_net")
sys.path.append("D:\Project\FGeo-ISRL\code\mcts_net")
sys.path.append("D:\Project\FGeo-ISRL\code\value_net")


# 导入必要的函数
from load_states_by_problem_id import load_states_by_problem_id
from load_sequence_by_problem_id import load_sequence_by_problem_id
from load_param_by_problem_id_and_theorem import load_param_by_problem_id_and_theorem
from load_final_state import load_final_state
from add_premise import add_premise
from JudgeSolveOrNot import solve_or_not

# ======================= 配置类 =======================
class PolicyConfig:
    model_name = "/home/lengmen/liyang/FGeoBDRL/Bert-base"
    model_weights = "/home/lengmen/liyang/FGeoBDRL/code/policy_net/pt/bert_base_test0.01/model_epoch100.pth"
    max_length = 256
    id_emb_dim = 64
    num_ids = 6980  # 训练时用的 id embedding size
    num_classes = 234
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValueConfig:
    model_name =  "/home/lengmen/liyang/FGeoBDRL/Bert-base"
    model_weights = "/home/lengmen/liyang/FGeoBDRL/code/value_net/pt/bert-base_test0.01/value_net_epoch100.pth"
    id_emb_dim = 64
    num_ids = 6980  # 保持和训练时一致
    max_length = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MCTSConfig:
    num_simulations = 100  # MCTS模拟次数
    c_puct = 1.5  # MCTS探索常数

# ======================= 策略网络模型 =======================
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

# ======================= 价值网络模型 =======================
class ValueNet(nn.Module):
    def __init__(self, model_name, num_ids, id_emb_dim):
        super(ValueNet, self).__init__()
        self.llm = AutoModel.from_pretrained(model_name)
        hidden = self.llm.config.hidden_size
        self.id_embedding = nn.Embedding(num_ids, id_emb_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden + id_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 回归在 [0, 1] 区间
        )

    def forward(self, input_ids, attention_mask, qid):
        output = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        id_emb = self.id_embedding(qid)
        combined = torch.cat((output, id_emb), dim=1)
        return self.regressor(combined).squeeze()

# ======================= MCTS节点类 =======================
class MCTSNode:
    def __init__(self, state, parent=None, action=None, param=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.param = param
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        
    def expand(self, actions, params):
        """扩展节点"""
        for action, param in zip(actions, params):
            child = MCTSNode(
                state=self.state,
                parent=self,
                action=action,
                param=param
            )
            self.children.append(child)
    
    def select(self, c_puct=1.0):
        """使用UCT公式选择子节点"""
        if not self.children:
            return self
            
        # 使用UCT公式: UCT = Q(v)/N(v) + c_puct * sqrt(ln(N(u))/N(v))
        ucb_values = []
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploit = child.value_sum / child.visits
            explore = c_puct * math.sqrt(math.log(self.visits) / child.visits)
            ucb_values.append(exploit + explore)
            
        return self.children[np.argmax(ucb_values)]
    
    def update(self, value):
        """更新节点统计信息"""
        self.visits += 1
        self.value_sum += value
        
    def is_leaf(self):
        """判断是否为叶节点"""
        return len(self.children) == 0

# ======================= MCTS搜索类 =======================
class MCTS:
    def __init__(self, num_simulations=50, c_puct=1.5):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        # 添加经验回放缓存
        self.experience_buffer = {}
        
    def get_valid_actions(self, problem_id, state):
        """获取有效的动作和参数"""
        valid_actions = []
        valid_params = []
        
        # 遍历所有可能的定理编号(0-234)
        for action in range(235):
            param_list = load_param_by_problem_id_and_theorem(problem_id, action)
            if param_list:
                if isinstance(param_list, list):
                    param = param_list[0]
                else:
                    param = param_list
                    
                try:
                    new_premise = add_premise(action, param)
                    if new_premise:
                        valid_actions.append(action)
                        valid_params.append(param)
                except:
                    continue
                    
        return valid_actions, valid_params
    
    def search(self, root_state, problem_id, final_state):
        """执行MCTS搜索"""
        # 检查经验回放缓存
        cache_key = (problem_id, root_state)
        if cache_key in self.experience_buffer:
            return self.experience_buffer[cache_key]
            
        root = MCTSNode(state=root_state)
        
        # 如果初始状态已经解决问题，直接返回
        if solve_or_not(final_state, root_state):
            return None, None
            
        # 获取有效的动作和参数
        valid_actions, valid_params = self.get_valid_actions(problem_id, root_state)
        
        if not valid_actions:
            return None, None
            
        # 扩展根节点
        root.expand(valid_actions, valid_params)
        
        # 执行模拟
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 选择阶段
            while not node.is_leaf():
                node = node.select(self.c_puct)
                search_path.append(node)
            
            # 如果叶节点已被访问，尝试扩展
            if node.visits > 0 and node is not root:
                try:
                    new_premise = add_premise(node.action, node.param)
                    if new_premise:
                        new_state = new_premise + "," + node.parent.state
                        node.state = new_state
                        
                        # 检查是否解决问题
                        if solve_or_not(final_state, new_state):
                            value = 1.0
                            for n in reversed(search_path):
                                n.update(value)
                            continue
                            
                        # 获取新的有效动作
                        valid_actions, valid_params = self.get_valid_actions(problem_id, new_state)
                        if valid_actions:
                            node.expand(valid_actions, valid_params)
                except:
                    value = 0.0
                    for n in reversed(search_path):
                        n.update(value)
                    continue
            
            # 模拟阶段
            if node.is_leaf() and node is not root:
                try:
                    new_premise = add_premise(node.action, node.param)
                    if new_premise:
                        new_state = new_premise + "," + node.parent.state
                        if solve_or_not(final_state, new_state):
                            value = 1.0
                        else:
                            value = 0.5
                    else:
                        value = 0.0
                except:
                    value = 0.0
            else:
                value = 0.5
                
            # 回溯更新
            for n in reversed(search_path):
                n.update(value)
        
        # 选择访问次数最多的动作
        if root.children:
            best_child = max(root.children, key=lambda n: n.visits)
            result = (best_child.action, best_child.param)
            
            # 将结果存入经验回放缓存
            self.experience_buffer[cache_key] = result
            
            return result
        
        return None, None

# ======================= 策略网络推理函数 =======================
def predict_top_k(model, tokenizer, state_text, id_text, config, k=1):
    """预测概率最高的 k 个动作类别"""
    model.eval()
    with torch.no_grad():
        qid_idx = hash(id_text) % config.num_ids
        qid_tensor = torch.tensor([qid_idx], dtype=torch.long).to(config.device)
        encoded = tokenizer(
            state_text,
            truncation=True,
            padding='max_length',
            max_length=config.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(config.device)
        attention_mask = encoded["attention_mask"].to(config.device)
        logits = model(input_ids, attention_mask, qid_tensor)
        # 获取前 k 个最高概率的动作类别及其概率
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k)
        # 转换为 Python 列表
        top_actions = top_indices.cpu().numpy().tolist()
        top_probs = top_probs.cpu().numpy().tolist()
        return list(zip(top_actions, top_probs))

# ======================= 价值网络推理函数 =======================
def predict_value(model, tokenizer, state_text, id_text, config):
    """预测状态的价值得分"""
    model.eval()
    with torch.no_grad():
        qid_idx = hash(id_text) % config.num_ids
        qid_tensor = torch.tensor([qid_idx], dtype=torch.long).to(config.device)

        encoded = tokenizer(
            state_text,
            truncation=True,
            padding='max_length',
            max_length=config.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(config.device)
        attention_mask = encoded["attention_mask"].to(config.device)

        value = model(input_ids, attention_mask, qid_tensor)
        return value.item()

def main():
    # 指定 JSON 文件存储路径
    json_file_path = "/home/lengmen/liyang/FGeoBDRL/code/tool/solve/result_p_v_mcts/results_with_mcts_and_value_net.json"
    # 新增：指定定理和参数记录的JSON文件路径
    theorem_params_path = "/home/lengmen/liyang/FGeoBDRL/code/tool/solve/result_p_v_mcts/theorem_params_record.json"
    
    # 初始化策略网络
    policy_config = PolicyConfig()
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_config.model_name)
    policy_model = PolicyNet(policy_config.model_name, policy_config.num_ids, policy_config.id_emb_dim, policy_config.num_classes).to(policy_config.device)
    policy_model.load_state_dict(torch.load(policy_config.model_weights, map_location=policy_config.device))
    
    # 初始化价值网络
    value_config = ValueConfig()
    value_tokenizer = AutoTokenizer.from_pretrained(value_config.model_name)
    value_model = ValueNet(value_config.model_name, value_config.num_ids, value_config.id_emb_dim).to(value_config.device)
    value_model.load_state_dict(torch.load(value_config.model_weights, map_location=value_config.device))
    
    # 初始化MCTS
    mcts_config = MCTSConfig()
    mcts = MCTS(num_simulations=mcts_config.num_simulations, c_puct=mcts_config.c_puct)
    
    print("✅ 策略网络、价值网络和MCTS加载完毕")
    
    results_detail = []  # 用于存储每个题目的求解结果
    # 新增：用于存储每个题目的定理和参数选择
    theorem_params_record = {}
    
    solved_count = 0
    unsolved_count = 0
    
    # 遍历题目ID范围 1 - 7001（含7001）
    for problem_id in range(1, 7000):
        # 简化输出，只显示当前正在解决的题目ID
        print(f"\r正在求解题目 {problem_id}/7001...", end="")
        
        # 加载题目数据
        state = load_states_by_problem_id(problem_id)
        sequence = load_sequence_by_problem_id(problem_id)
        final_state = load_final_state(problem_id)
        
        # 如果状态或目标状态不存在，则跳过此题
        if not state or not final_state:
            continue

        # 记录求解信息
        steps = 0
        max_steps = 30  # 限制最大求解步数
        consecutive_zero_count = 0  # 记录连续三次最高预测为 0 的次数
        solved_by_rule = False   # 标记是否因连续三次预测 0 而提前解题
        value_improvements = []  # 记录每一步的价值提升
        action_sources = []      # 记录每一步动作的来源（MCTS或策略网络）
        # 新增：记录该题目的定理和参数选择
        theorem_params_steps = []

        # 如果初始状态已满足目标，则认为题目已解出（步数为 0）
        if solve_or_not(final_state, state):
            solved = True
        else:
            # 自动求解循环
            while steps < max_steps and not solve_or_not(final_state, state):
                steps += 1
                
                # 求解过程中再次检查是否达到目标状态
                if solve_or_not(final_state, state):
                    break
                
                # 使用策略网络预测最佳动作
                state_for_prediction = state.strip()
                id_for_prediction = str(problem_id)
                
                # 评估当前状态的价值
                current_value = predict_value(value_model, value_tokenizer, state_for_prediction, id_for_prediction, value_config)
                
                # 获取策略网络预测的动作
                policy_actions = predict_top_k(policy_model, policy_tokenizer, state_for_prediction, id_for_prediction, policy_config, k=5)
                
                # 检查最高预测是否为 0
                if policy_actions and policy_actions[0][0] == 0:
                    consecutive_zero_count += 1
                else:
                    consecutive_zero_count = 0
                
                if consecutive_zero_count >= 3:
                    solved_by_rule = True
                    # 新增：记录提前解题
                    theorem_params_steps.append({
                        "step": steps,
                        "action": 0,  # 表示提前解题
                        "param": None,
                        "source": "Rule"
                    })
                    break
                
                # 使用MCTS搜索最佳动作
                mcts_action, mcts_param = mcts.search(
                    root_state=state_for_prediction,
                    problem_id=problem_id,
                    final_state=final_state
                )
                
                # 尝试应用MCTS搜索的动作
                mcts_value = None
                if mcts_action is not None:
                    try:
                        new_premise = add_premise(mcts_action, mcts_param)
                        if new_premise:
                            mcts_state = new_premise + "," + state
                            mcts_value = predict_value(value_model, value_tokenizer, mcts_state, id_for_prediction, value_config)
                    except:
                        pass
                
                # 尝试应用策略网络预测的动作
                policy_value = None
                policy_action = None
                policy_param = None
                policy_state = None
                
                for action_idx, (theorem, prob) in enumerate(policy_actions):
                    # 再次检查目标是否已经达到
                    if solve_or_not(final_state, state):
                        break
                    
                    param_list = load_param_by_problem_id_and_theorem(problem_id, theorem)
                    if not param_list:
                        continue
                    param = param_list[0] if isinstance(param_list, list) else param_list
                    
                    try:
                        # 模拟应用定理
                        new_premise = add_premise(theorem, param)
                        if new_premise:
                            # 模拟新状态
                            simulated_state = new_premise + "," + state
                            
                            # 评估模拟状态的价值
                            simulated_value = predict_value(value_model, value_tokenizer, simulated_state, id_for_prediction, value_config)
                            
                            # 只有当价值提升时才考虑
                            if simulated_value > current_value:
                                policy_value = simulated_value
                                policy_action = theorem
                                policy_param = param
                                policy_state = simulated_state
                                break
                    except:
                        continue
                
                # 比较MCTS和策略网络的结果，选择价值更高的
                action_found = False
                if mcts_value is not None and policy_value is not None:
                    # 两者都有有效结果，选择价值更高的
                    if mcts_value >= policy_value:
                        state = mcts_state
                        value_improvements.append((current_value, mcts_value))
                        action_sources.append("MCTS")
                        # 新增：记录选择的定理和参数
                        theorem_params_steps.append({
                            "step": steps,
                            "action": mcts_action,
                            "param": mcts_param,
                            "source": "MCTS",
                            "value_improvement": mcts_value - current_value
                        })
                        action_found = True
                    else:
                        state = policy_state
                        value_improvements.append((current_value, policy_value))
                        action_sources.append("Policy")
                        # 新增：记录选择的定理和参数
                        theorem_params_steps.append({
                            "step": steps,
                            "action": policy_action,
                            "param": policy_param,
                            "source": "Policy",
                            "value_improvement": policy_value - current_value
                        })
                        action_found = True
                elif mcts_value is not None:
                    # 只有MCTS有有效结果
                    if mcts_value > current_value:
                        state = mcts_state
                        value_improvements.append((current_value, mcts_value))
                        action_sources.append("MCTS")
                        # 新增：记录选择的定理和参数
                        theorem_params_steps.append({
                            "step": steps,
                            "action": mcts_action,
                            "param": mcts_param,
                            "source": "MCTS",
                            "value_improvement": mcts_value - current_value
                        })
                        action_found = True
                elif policy_value is not None:
                    # 只有策略网络有有效结果
                    state = policy_state
                    value_improvements.append((current_value, policy_value))
                    action_sources.append("Policy")
                    # 新增：记录选择的定理和参数
                    theorem_params_steps.append({
                        "step": steps,
                        "action": policy_action,
                        "param": policy_param,
                        "source": "Policy",
                        "value_improvement": policy_value - current_value
                    })
                    action_found = True
                
                if not action_found:
                    break
            
            # 判断题目是否解出：满足目标状态或触发提前解题规则均视为解出
            solved = solve_or_not(final_state, state) or solved_by_rule
        
        if solved:
            solved_count += 1
        else:
            unsolved_count += 1
        
        # 每100题显示一次进度统计
        if problem_id % 100 == 0:
            print(f"\n已完成 {problem_id}/7001 题，已解出: {solved_count}，未解出: {unsolved_count}")
        
        # 保存该题目的结果信息
        results_detail.append({
            "problem_id": problem_id,
            "solved": solved,
            "steps": steps,
            "value_improvements": value_improvements,
            "action_sources": action_sources
        })
        
        # 新增：保存该题目的定理和参数选择记录
        theorem_params_record[str(problem_id)] = {
            "solved": solved,
            "steps": steps,
            "theorem_params": theorem_params_steps
        }
    
    # 汇总统计信息
    summary = {
        "solved_count": solved_count,
        "unsolved_count": unsolved_count,
        "total_attempted": solved_count + unsolved_count
    }
    
    final_result = {
        "summary": summary,
        "details": results_detail
    }
    
    # 写入 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)
    
    # 新增：写入定理和参数记录的JSON文件
    with open(theorem_params_path, 'w', encoding='utf-8') as f:
        json.dump(theorem_params_record, f, ensure_ascii=False, indent=4)
    
    print(f"\n\n求解完成！总共尝试 {solved_count + unsolved_count} 题，解出 {solved_count} 题，未解出 {unsolved_count} 题")
    print(f"结果已保存至：{json_file_path}")
    print(f"定理和参数记录已保存至：{theorem_params_path}")

if __name__ == "__main__":
    main()