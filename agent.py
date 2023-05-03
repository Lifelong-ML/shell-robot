import torch
import glob
import torch.optim as optim
from pathlib import Path
from dataloaders import PickleLoader
from models import ResNetUNet
import argparse
import tqdm
from typing import List
import numpy as np

from metric import Metric


class Agent:

    def __init__(self, agent_idx: int, task_list: List[Path], num_epochs: int):
        self.agent_idx = agent_idx
        self.task_list = task_list
        self.num_epochs = num_epochs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNetUNet(3, 3).to(self.device)

    def learn(self, task_idx: int) -> List[float]:
        """
        train a task
        """
        assert 0 <= task_idx < len(
            self.task_list
        ), f"task_idx out of range. Must be between 0 and {len(self.task_list) - 1}. Got {task_idx}"

        data_path = self.task_list[task_idx]
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # Load data
        dataset = PickleLoader(data_path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=2,
                                                 shuffle=True)

        # Define the number of epochs

        running_loss_list = []

        # Train loop
        pbar = tqdm.tqdm(range(self.num_epochs),
                         desc=f'Agent {self.agent_idx} {data_path.name} Epoch')
        for epoch in pbar:
            running_loss = 0.0
            total_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # Get input and target from the dataset
                inputs, targets = data['input'].to(
                    self.device), data['target'].to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Output image is Bx3xNxN, target is BxNxN. Output and target must be flattened
                # across their spacial axis to be treated as a per-pixel classification problem.
                B, C, _, _ = outputs.shape
                outputs = outputs.reshape(B, C, -1)
                targets = targets.reshape(B, -1)

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # Print running loss
                running_loss += loss.item()
                total_loss += loss.item()
                running_loss_list.append(running_loss / len(data))
                if i % 1000 == 999:  # Print every 1000 mini-batches
                    print(
                        f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}"
                    )
                    running_loss = 0.0
            # Update tqdm bar with latest loss
            pbar.set_postfix({'loss': total_loss / len(dataloader)})
        return running_loss_list

    @torch.no_grad()
    def evaluate(self, task_idx: int) -> Metric:
        """
        evaluate a task
        """
        assert 0 <= task_idx < len(
            self.task_list
        ), f"task_idx out of range. Must be between 0 and {len(self.task_list) - 1}. Got {task_idx}"
        data_path = self.task_list[task_idx]
        criterion = torch.nn.CrossEntropyLoss()
        # Load data
        dataset = PickleLoader(data_path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False)

        pixel_accuracies = []
        calibration_errors = []
        losses = []
        for data in dataloader:
            inputs, targets = data['input'].to(self.device), data['target'].to(
                self.device)
            outputs = self.model(inputs)

            B, C, _, _ = outputs.shape

            # Flatten outputs and targets to get rid of spacial dimensions.
            outputs = outputs.reshape(B, C, -1)
            targets = targets.reshape(B, -1)

            # This is an accuracy measure between 0 and 1 across all pixels.
            correct_pixels = (outputs.argmax(dim=1) == targets).sum()
            total_pixels = targets.numel()
            pixel_accuracy = correct_pixels / total_pixels

            # Measure calibration error by computing the average confidence of the correct class.
            probability_of_correct_class = outputs.softmax(dim=1).gather(
                1, targets.unsqueeze(1)).squeeze(1)
            calibration_error = 1 - probability_of_correct_class.mean()

            pixel_accuracies.append(pixel_accuracy.item())
            calibration_errors.append(calibration_error.item())

            # Compute loss
            loss = criterion(outputs, targets)
            losses.append(loss.item())

        return Metric(loss=np.mean(losses),
                      pixel_average_accuracy=np.mean(pixel_accuracies),
                      calibration_error=np.mean(calibration_errors))

    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()],
                         -1)

    def load_weights(self, weights):
        # weights is a vector
        beg = 0
        for p in self.model.parameters():
            p.data.copy_(weights[beg:beg + p.numel()].reshape(*p.data.shape))
            beg += p.numel()
