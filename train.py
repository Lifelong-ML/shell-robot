import torch
import torch.optim as optim
from pathlib import Path
from dataloaders import PickleLoader
from models import ResNetUNet
import argparse
import tqdm

# Get path to data from command line
parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=Path, help='Processed data path.')
parser.add_argument('--checkpoint_dir',
                    type=Path,
                    default="./model_checkpoints",
                    help='Path to save model checkpoints.')
args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
checkpoint_dir.mkdir(exist_ok=True)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model and move it to device
model = ResNetUNet(3, 3).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Load data
dataset = PickleLoader(args.data_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Define the number of epochs
num_epochs = 1000

# Train loop
for epoch in tqdm.tqdm(range(num_epochs), desc='Epoch'):
    running_loss = 0.0
    total_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # Get input and target from the dataset
        inputs, targets = data['input'].to(device), data['target'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

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
    print("Epoch loss: ", total_loss / len(dataloader))

    # Save model checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(),
                checkpoint_dir / f'model_epoch_{epoch+1:06d}.pth')

print("Training finished.")