import torch
import torch.optim as optim
from pathlib import Path
from dataloaders import PickleLoader
from models import ResNetUNet
import argparse
import tqdm
import os
from agent import Agent

# Get path to data from command line
parser = argparse.ArgumentParser()

parser.add_argument('data_path', type=Path, help='Processed data path to the problem folder (including all tasks, tasks should named as task1, task2...')
parser.add_argument('--checkpoint_dir',
                    type=Path,
                    default="./model_checkpoints",
                    help='Path to save model checkpoints.')
parser.add_argument('--num_tasks',
                    type=int,
                    help='number of tasks')
parser.add_argument('--num_agents',
                    type=int,
                    help='number of agents')
args = parser.parse_args()


checkpoint_dir = args.checkpoint_dir
checkpoint_dir.mkdir(exist_ok=True)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Metric:
    def __init__(self):
        self.loss = 0.0
        self.mAP = 0.0

    def set(self, loss, mAP):
        self.loss = loss
        self.mAP = mAP 

agents = [Agent(args.num_tasks) for _ in range(args.num_agents)]
metrics = [[Metric() for i in range(args.num_tasks)] for j in range(args.num_tasks)]


for task in range(args.num_tasks):
    data_path = Path(os.path.join(args.data_path, f"task{task+1}"))

    # training
    theta = 0.0
    for agent in agents:
        agent.learn(data_path)
        theta += agent.get_weights().data
    avg_theta = theta / args.num_agents
    for agent in agents: agent.load_weights(avg_theta)

    # testing
    for prev_task in range(task+1):
        prev_data_path = Path(os.path.join(args.data_path, f"task{prev_task+1}"))
        metric = agents[0].evaluate(prev_data_path, Metric())
        metrics[task][prev_task] = metric
    print(f"[info] Finishing task {task+1}")

    torch.save(agents[0].model.state_dict(), os.path.join(checkpoint_dir, f"model_weights_task{task+1}.pth"))
    torch.save(metrics, os.path.join(checkpoint_dir, f"metrics.pth"))
