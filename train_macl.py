import torch
import torch.optim as optim
from pathlib import Path
from dataloaders import PickleLoader
from models import ResNetUNet
import argparse
import tqdm
from agent import Agent
from metric import Metric
import numpy as np

# Get path to data from command line
parser = argparse.ArgumentParser()

parser.add_argument(
    'data_path',
    type=Path,
    help=
    'Processed data path to the problem folder (including all tasks, tasks should named as task1, task2...'
)
parser.add_argument('--checkpoint_dir',
                    type=Path,
                    default="./model_checkpoints",
                    help='Path to save model checkpoints.')
parser.add_argument('--num_tasks', type=int, help='number of tasks')
parser.add_argument('--num_agents', type=int, help='number of agents')
parser.add_argument('--num_epochs', type=int, help='number of epochs')
parser.add_argument('--single_agent_task_id',
                    type=int,
                    default=None,
                    help='single task expert mode')
args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
checkpoint_dir.mkdir(exist_ok=True)

assert args.data_path.is_dir(), f"{args.data_path} is not a directory."
assert args.num_tasks > 0, f"num_tasks must be greater than 0. Got {args.num_tasks}"
assert args.num_agents > 0, f"num_agents must be greater than 0. Got {args.num_agents}"
assert args.num_epochs > 0, f"num_epochs must be greater than 0. Got {args.num_epochs}"

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_task_matrix(num_tasks: int, num_agents: int, int_offset: int = 0):
    task_folders = [(args.data_path / f"task_{task_id + int_offset + 1:03d}")
                    for task_id in range(num_tasks * num_agents)]

    for task_folder in task_folders:
        assert task_folder.is_dir(), f"{task_folder} is not a directory."
        assert len(
            list(task_folder.glob("*.pkl"))
        ) > 0, f"task folder {task_folder} does not contain any .pkl files."

    return np.array(task_folders).reshape(args.num_agents, args.num_tasks)


if args.single_agent_task_id is not None:
    print("[info] Running in single agent expert mode for agent",
          args.single_agent_task_id)
    task_matrix = setup_task_matrix(1, 1, args.single_agent_task_id)
else:
    task_matrix = setup_task_matrix(args.num_tasks, args.num_agents)

print(
    f"[info] Training {args.num_agents} agents on unique tasks sets of length {args.num_tasks}."
)

agents = [
    Agent(agent_idx, task_lst, args.num_epochs)
    for agent_idx, task_lst in enumerate(task_matrix)
]

for task_set_idx in range(args.num_tasks):

    # Train each model separately
    trained_weights = []
    for agent_idx, agent in enumerate(agents):
        agent.learn(task_set_idx)
        trained_weights.append(agent.get_weights().data)
        torch.save(
            agent.model.state_dict(), checkpoint_dir /
            f"agent_{agent_idx:03d}_model_weights_after_task_set_idx_{task_set_idx:03d}.pth"
        )

    # Average weights
    average_weights = torch.mean(torch.stack(trained_weights), dim=0)
    for agent in agents:
        agent.load_weights(average_weights)

    agent_prev_task_performances = {
        agent_idx: {}
        for agent_idx in range(args.num_agents)
    }
    # Test averaged weights on all previous tasks, and the current task
    for prev_task_idx in tqdm.tqdm(range(task_set_idx + 1),
                                   desc="Task Set Index"):
        for agent_idx in range(args.num_agents):
            agent_prev_task_perf = agents[agent_idx].evaluate(prev_task_idx)
            agent_prev_task_performances[agent_idx][
                prev_task_idx] = agent_prev_task_perf

    torch.save(
        agents[0].model.state_dict(), checkpoint_dir /
        f"average_model_weights_after_task_set_idx_{task_set_idx:03d}.pth")
    torch.save(
        agent_prev_task_performances, checkpoint_dir /
        f"perf_metrics_after_task_set_idx_{task_set_idx:03d}.pth")

    print(f"[info] Finishing task set idx {task_set_idx}")
