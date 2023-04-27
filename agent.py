import torch
import glob
import torch.optim as optim
from pathlib import Path
from dataloaders import PickleLoader
from models import ResNetUNet
import argparse
import tqdm


class Agent:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.model = ResNetUNet(3, 3).to(self.device)
        self.task = 0

    def learn(self, data_path): 
        """
        train a task
        """
        self.task += 1
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # Load data
        dataset = PickleLoader(data_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        # Define the number of epochs
        num_epochs = 10

        # Train loop
        for epoch in tqdm.tqdm(range(num_epochs), desc='Epoch'):
            running_loss = 0.0
            total_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # Get input and target from the dataset
                inputs, targets = data['input'].to(self.device), data['target'].to(self.device)

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
                if i % 1000 == 999:  # Print every 1000 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
                    running_loss = 0.0
            print(f"Task {self.task:3d} Epoch loss: {total_loss /len(dataloader):.2f}")

    @torch.no_grad()
    def evaluate(self, data_path, metric):
        """
        evaluate a task
        """
        criterion = torch.nn.CrossEntropyLoss()
        # Load data
        dataset = PickleLoader(data_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

        total_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, targets = data['input'].to(self.device), data['target'].to(self.device)
            outputs = self.model(inputs)
            B, C, _, _ = outputs.shape
            outputs = outputs.reshape(B, C, -1)
            targets = targets.reshape(B, -1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        metric.set(total_loss/len(dataloader), 0.0)

    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()], -1)
        
    def load_weights(self, weights):
        # weights is a vector
        beg = 0
        for p in self.model.parameters():
            p.data.copy_(weights[beg:beg+p.numel()].reshape(*p.data.shape))
            beg += p.numel()
