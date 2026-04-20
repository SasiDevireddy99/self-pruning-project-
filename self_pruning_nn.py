import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Part 1: The Prunable Linear Layer
# ---------------------------------------------------------
class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns to prune itself.
    Each weight has an associated learnable gate parameter.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate scores (unconstrained) - same shape as weight
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Standard Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize bias to zero
        nn.init.constant_(self.bias, 0)
        # Initialize gate scores to a positive value so that initially gates are open (~0.88 with sigmoid(2.0))
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, x):
        # Transform gate_scores to gates (0 to 1) using Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Prune weights element-wise
        pruned_weights = self.weight * gates
        
        # Return standard linear operation output
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity(self, threshold=1e-2):
        """Calculates the sparsity of this layer based on gate values."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            pruned_count = (gates < threshold).sum().item()
            total_count = gates.numel()
            return pruned_count, total_count

# ---------------------------------------------------------
# Defining the Model
# ---------------------------------------------------------
class SelfPruningMLP(nn.Module):
    def __init__(self, input_size=3072, hidden_size=512, num_classes=10):
        super(SelfPruningMLP, self).__init__()
        self.fc1 = PrunableLinear(input_size, hidden_size)
        self.fc2 = PrunableLinear(hidden_size, hidden_size // 2)
        self.fc3 = PrunableLinear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        # Flatten image
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_total_sparsity(self, threshold=1e-2):
        total_pruned = 0
        total_weights = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                p, t = module.get_sparsity(threshold)
                total_pruned += p
                total_weights += t
        return (total_pruned / total_weights) * 100 if total_weights > 0 else 0

    def get_sparsity_loss(self):
        """Calculates the L1 penalty on all gate values across the network."""
        sparsity_loss = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                # Sigmoid gates are always positive, so L1 is just the sum
                sparsity_loss += torch.sum(torch.sigmoid(module.gate_scores))
        return sparsity_loss

# ---------------------------------------------------------
# Part 3: Training and Evaluation Logic
# ---------------------------------------------------------
def train(model, train_loader, optimizer, lambda_reg, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        
        # Classification Loss
        ce_loss = F.cross_entropy(output, target)
        
        # Sparsity Regularization Loss
        sparsity_loss = model.get_sparsity_loss()
        
        # Total Loss
        loss = ce_loss + lambda_reg * sparsity_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Tracking accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tCE Loss: {ce_loss.item():.6f}\tSparsity Loss: {sparsity_loss.item():.6f}')

    accuracy = 100. * correct / total
    return total_loss / len(train_loader), accuracy

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def run_experiment(lambdas, epochs=10):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # CIFAR-10 setup
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    results = []

    for l_val in lambdas:
        print(f"\n{'='*50}")
        print(f"Running Experiment with Lambda = {l_val}")
        print(f"{'='*50}")
        
        model = SelfPruningMLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        best_acc = 0
        final_sparsity = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(model, train_loader, optimizer, l_val, epoch)
            test_loss, test_acc = test(model, test_loader)
            sparsity = model.get_total_sparsity()
            
            print(f'Epoch {epoch}: Test Acc: {test_acc:.2f}%, Sparsity: {sparsity:.2f}%')
            
            if test_acc > best_acc:
                best_acc = test_acc
            final_sparsity = sparsity

        results.append({
            'lambda': l_val,
            'test_accuracy': best_acc,
            'sparsity': final_sparsity,
            'model': model # Store model for later visualization
        })

    return results

def plot_gate_distribution(model, lambda_val):
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Gate Values (Lambda={lambda_val})')
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    plot_path = f'results/gate_dist_lambda_{lambda_val}.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Test lambdas: Low, Medium, High
    lambda_list = [1e-6, 1e-5, 5e-5] 
    
    # We will use fewer epochs for demonstration purposes if running in restricted environments,
    # but 10-15 is usually needed for noticeable pruning.
    results = run_experiment(lambda_list, epochs=5)
    
    # Summarize results
    print("\nSummary Results:")
    print("-" * 50)
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity (%)':<15}")
    print("-" * 50)
    for res in results:
        print(f"{res['lambda']:<10} | {res['test_accuracy']:<20.2f} | {res['sparsity']:<15.2f}")
    
    # Plot for the "best" model (balancing accuracy and sparsity)
    # Usually the middle lambda or the one with highest accuracy if sparsity is reasonable.
    best_res = results[1] # Using the medium lambda for the plot
    plot_gate_distribution(best_res['model'], best_res['lambda'])
