import torch
import numpy as np
import math
import logging
from tqdm import tqdm
import pandas as pd
import sys
import os
import json

sys.path.append("D:\Project\FGeo-ISRL\code\policy_net")
sys.path.append("D:\Project\FGeo-ISRL\code\value_net")
from add_premise import add_premise
from JudgeSolveOrNot import solve_or_not
from load_param_by_problem_id_and_theorem import load_param_by_problem_id_and_theorem
from load_final_state import load_final_state

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# num_simulations and max_steps are important

class PureMCTSNode:
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
            child = PureMCTSNode(
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

class PureMCTS:
    def __init__(self, num_simulations=300, c_puct=1.5, experience_path=None):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.experience_buffer = {}  # 经验回放缓存
        
        # 如果提供了经验文件路径，则加载经验
        if experience_path and os.path.exists(experience_path):
            self.load_experience(experience_path)
        
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
            
        root = PureMCTSNode(state=root_state)
        
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
            if node.visits > 0 and not node is root:
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
            if node.is_leaf() and not node is root:
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
    
    def save_experience(self, path):
        """保存经验回放缓存到文件"""
        # 将经验缓存转换为可序列化的格式
        serializable_buffer = {}
        for (problem_id, state), (action, param) in self.experience_buffer.items():
            key = f"{problem_id}|||{state}"  # 使用分隔符合并键
            if action is not None and param is not None:
                serializable_buffer[key] = (int(action), param)
        
        # 保存到文件
        with open(path, 'w') as f:
            json.dump(serializable_buffer, f)
        
        logging.info(f"经验回放缓存已保存到: {path}")
    
    def load_experience(self, path):
        """从文件加载经验回放缓存"""
        try:
            with open(path, 'r') as f:
                serializable_buffer = json.load(f)
            
            # 将序列化格式转换回原始格式
            for key, value in serializable_buffer.items():
                parts = key.split("|||", 1)
                if len(parts) == 2:
                    problem_id, state = parts
                    self.experience_buffer[(int(problem_id), state)] = value
            
            logging.info(f"已加载经验回放缓存，共 {len(self.experience_buffer)} 条记录")
        except Exception as e:
            logging.error(f"加载经验回放缓存失败: {e}")
            self.experience_buffer = {}

def predict_with_mcts(test_data_path, output_path, experience_path=None, save_experience=True):
    """使用纯MCTS进行预测并实时显示解题进度"""
    # 加载测试数据
    test_data = pd.read_csv(test_data_path)
    
    # 获取唯一的题目ID列表
    unique_problem_ids = test_data['id'].unique()
    total_problems = len(unique_problem_ids)
    
    # 对每个唯一题目ID，找到其初始状态（最短的state）
    problem_initial_states = {}
    for problem_id in unique_problem_ids:
        problem_rows = test_data[test_data['id'] == problem_id]
        # 找到最短的state作为初始状态
        initial_state = min(problem_rows['state'].tolist(), key=len)
        problem_initial_states[problem_id] = initial_state
    
    predictions = []
    
    # 统计变量
    solved_problems = 0
    unsolved_problems = 0
    
    # 初始化MCTS，加载经验回放
    mcts = PureMCTS(num_simulations=100, experience_path=experience_path)
    
    # 对每个唯一的问题进行预测
    for problem_id in tqdm(unique_problem_ids, desc="预测进度"):
        initial_state = problem_initial_states[problem_id]
        final_state = load_final_state(problem_id)
        
        if final_state:
            # 检查初始状态是否已经解决问题
            if solve_or_not(final_state, initial_state):
                solved_problems += 1
                predictions.append({
                    "problem_id": problem_id,
                    "predicted_action": None,
                    "predicted_param": None,
                    "initial_state": initial_state,
                    "final_state": final_state,
                    "solved": True,
                    "steps": 0
                })
                # 实时打印解题进度
                print(f"\r当前题目: {problem_id}, 已解出: {solved_problems}, 未解出: {unsolved_problems}, 总计: {total_problems}", end="")
                continue
            
            # 执行MCTS搜索
            current_state = initial_state
            steps = 0
            max_steps = 30  # 限制最大步数
            solved = False
            action_history = []
            param_history = []
            
            while steps < max_steps and not solved:
                predicted_action, predicted_param = mcts.search(
                    root_state=current_state,
                    problem_id=problem_id,
                    final_state=final_state
                )
                
                # 如果没有找到有效动作，终止搜索
                if predicted_action is None:
                    break
                
                # 记录动作历史
                action_history.append(predicted_action)
                param_history.append(predicted_param)
                
                # 应用动作
                try:
                    new_premise = add_premise(predicted_action, predicted_param)
                    if new_premise:
                        new_state = new_premise + "," + current_state
                        current_state = new_state
                        steps += 1
                        
                        # 检查是否解决问题
                        if solve_or_not(final_state, current_state):
                            solved = True
                            break
                    else:
                        break
                except:
                    break
            
            # 更新统计信息
            if solved:
                solved_problems += 1
            else:
                unsolved_problems += 1
            
            predictions.append({
                "problem_id": problem_id,
                "predicted_action": action_history,
                "predicted_param": param_history,
                "initial_state": initial_state,
                "final_state": final_state,
                "current_state": current_state,
                "solved": solved,
                "steps": steps
            })
            
            # 实时打印解题进度
            print(f"\r当前题目: {problem_id}, 已解出: {solved_problems}, 未解出: {unsolved_problems}, 总计: {total_problems}", end="")
    
    # 计算解题率
    solve_rate = solved_problems / total_problems if total_problems > 0 else 0
    
    # 输出最终统计信息
    print("\n\n解题统计:")
    print(f"总题目数: {total_problems}")
    print(f"解出题目数: {solved_problems} ({solve_rate:.2%})")
    print(f"未解出题目数: {unsolved_problems} ({1-solve_rate:.2%})")
    
    # 保存预测结果
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    logging.info(f"预测结果已保存到: {output_path}")
    
    # 保存经验回放缓存
    if save_experience:
        if experience_path is None:
            experience_path = "D:\Project\FGeo-ISRL\code\mcts_net\mcts_experience.json"
        mcts.save_experience(experience_path)
    
    return solved_problems, unsolved_problems, total_problems

# 简化后的主函数，一键运行
def main():
    # 尝试加载配置文件
    config_path = "D:\Project\FGeo-ISRL\code\mcts_net\config.json"
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"已加载配置文件: {config_path}")
        else:
            # 默认配置
            config = {
                "test_data_path": "D:\Project\FGeo-ISRL\code\mcts_net\dataset_for_mcts.csv",
                "output_path": "D:\Project\FGeo-ISRL\code\mcts_net\mcts_predictions.csv",
                "experience_path": "D:\Project\FGeo-ISRL\code\mcts_net\mcts_experience.json",
                "num_simulations": 100,
                "max_steps": 30
            }
            print("未找到配置文件，使用默认配置")
    except Exception as e:
        print(f"加载配置文件失败: {e}，使用默认配置")
        # 默认配置
        config = {
            "test_data_path": "D:\Project\FGeo-ISRL\code\mcts_net\dataset_for_mcts.csv",
                "output_path": "D:\Project\FGeo-ISRL\code\mcts_net\mcts_predictions.csv",
                "experience_path": "D:\Project\FGeo-ISRL\code\mcts_net\mcts_experience.json",
                "num_simulations": 5,
                "max_steps": 20
        }
    
    print("开始MCTS解题评估...")
    print(f"使用经验回放文件: {config['experience_path']}")
    
    # 检查经验回放文件是否存在
    if os.path.exists(config['experience_path']):
        print(f"发现已有经验回放文件，将加载已有经验")
    else:
        print(f"未找到经验回放文件，将创建新的经验回放")
    
    # 执行MCTS预测
    predict_with_mcts(
        test_data_path=config['test_data_path'],
        output_path=config['output_path'],
        experience_path=config['experience_path'],
        save_experience=True
    )

if __name__ == "__main__":
    main()