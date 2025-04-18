import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import to_pil_image
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re

# packages for optuna
import optuna

# Training function
def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*20}')
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train_set)
        epoch_acc = running_corrects.double() / len(train_set)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return model

# Model evaluation
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Confusion matrix: [[TN, FP], [FN, TP]]
    print(all_labels)
    # labels=['HF','C']
    # positive class = control
    # negative class = HF

    # labels = [0,1]
    # positive class = 1 = HF
    # negative class = 0 = control
    cm = confusion_matrix(all_labels, all_preds,labels=[0,1])
    TN, FP, FN, TP = cm.ravel()

    # Precision, Recall, False Positive Rate
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    false_positive_rate = FP / (FP + TN) if FP + TN > 0 else 0.0

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'False Positive Rate: {false_positive_rate:.4f}')
    
    
def show_gradcam_side_by_side(input_tensor, heatmap, saveFileName):
    """
    input_tensor: torch.Tensor of shape [3, H, W]
    heatmap: numpy array of shape [H, W], normalized to [0,1]
    """
    # Remove batch dim if needed
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.squeeze(0)

    # Convert to PIL image
    input_img = to_pil_image(input_tensor.cpu())

    # Squeeze heatmap to 2D
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.squeeze().cpu().numpy()
    else:
        heatmap = np.squeeze(heatmap)

    # Normalize and scale to [0, 255]
    heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)))

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(input_img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(heatmap, cmap='jet')
    axs[1].set_title("Grad-CAM Mask")
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(saveFileName)
    
# Hyperparameter optimization using optuna
# Function adapted from: https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_int('batch_size', 16, 64)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(in_features, 2)  # CIFAR-10 has 10 classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Training loop
    model.train()
    for epoch in range(10):  # Adjust number of epochs as needed
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


##### main #####
test_animal_sets = np.array([[1,6, 27, 29, 31, 34]])#[5,9],[1,6],[7,4],[3,8],[2,6]]) 13, 15, 17, 19, 21, 23, 25,
[num_folds, cols] = test_animal_sets.shape

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms (including normalization)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])   # ImageNet std
])

# Load the full dataset
data_dir = '//lustre//ogunnaike//users//2420//matlab_example//NZ-physiology-data//PESimages//CO_CoBF_30m_48slices//'

# Test different sets of animals for train/test split
for fold, test_animal_ids in enumerate(test_animal_sets):
    print(f'\n\n=== Fold {fold + 1} | Test Animals: {test_animal_ids} ===\n')
    
    
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    # Extract animal ID from filename (e.g., A5, A9)
    def extract_animal_id(path):
        filename = os.path.basename(path)
        match = re.search(r'A(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    # Map each image to its animal ID
    animal_ids = [extract_animal_id(fp[0]) for fp in full_dataset.samples]
    
    # Indices for train and test based on animal ID
    train_indices = [i for i, aid in enumerate(animal_ids) if aid not in test_animal_ids]
    test_indices = [i for i, aid in enumerate(animal_ids) if aid in test_animal_ids]
    
    # Create Subsets and Dataloaders
    batch_size = 32
    train_set = Subset(full_dataset, train_indices)
    test_set = Subset(full_dataset, test_indices)
    
    """
    # Print file names in train_set
    print("Train set file names:")
    for idx in train_set.indices:
        print(full_dataset.samples[idx][0])
    
    # Print file names in test_set
    print("\nTest set file names:")
    for idx in test_set.indices:
        print(full_dataset.samples[idx][0])
    """
    
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print('Training and test sets split')
    
    # Class names
    class_names = full_dataset.classes
    
    # Load the ResNet model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # binary classification
    model_ft = model_ft.to(device)
    
    # Print model structure and number of parameters
    total_params = 0
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.numel()} params')
            total_params += param.numel()
    
    print(f'Total trainable parameters: {total_params}')
    
    # Loss, optimizer, and LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    print('Model loaded')


    # Train the model
    print('Training model...')
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader, num_epochs=3)
    
    
    # Evaluate model
    evaluate_model(model_ft, test_loader)
    
    # Extract features used to train model
    # Initialize GradCAM on the last conv layer
    cam_extractor = GradCAM(model_ft, target_layer='layer4')
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Load your image
    img_path = '//lustre//ogunnaike//users//2420//matlab_example//NZ-physiology-data//PESimages//CO_CoBF_30m_48slices//C//CO_CoBF_30m_A8_S18.png'
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dim
    
    # Forward pass
    out = model_ft(input_tensor)
    
    # Get class index with highest score
    class_idx = out.squeeze().argmax().item()
    
    # Extract CAM
    activation_map = cam_extractor(class_idx, out)
    
    # Overlay on image
    result = overlay_mask(to_pil_image(input_tensor.squeeze()), to_pil_image(activation_map[0], mode='F'), alpha=0.5)

    # Show result
    plt.imshow(result)
    plt.title(f'GradCAM for class {class_idx}')
    plt.axis('off')
    saveFileName = 'GradCAM_overlay_CO_CoBF_30m_A8_S18.png'
    plt.savefig(saveFileName)
    plt.close()
    
    # Visualize
    saveFile = 'GradCAM_CO_CoBF_30m_A8_S18.png'
    show_gradcam_side_by_side(input_tensor, activation_map, saveFile)
    


