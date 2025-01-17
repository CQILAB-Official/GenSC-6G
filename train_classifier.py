import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import argparse
from allmodels import InceptionV3, QInceptionV3, ResNet50, QResNet50, ViT32, QViT32, SwinT, QSwinT
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, auc, recall_score, precision_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Pass through params
def parse_args():
    parser = argparse.ArgumentParser(description="Pass through parameters")
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='[inceptionv3, qinceptionv3, resnet, \
        qresnet, vit, qvit, swin, qswin]')
    parser.add_argument('--dataset', type=str, default='aider', help='[aider, euro, hrsc2016]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--load_model', type=str, default=None, help='Folder location')
    parser.add_argument('--noise_type', type=str, default='awgn', help='[awgn, rayleigh]')
    parser.add_argument('--device', type=int, default=0, help='[0, 1]')
    return parser.parse_args()

# Parse arguments
args = parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
root_folder = "testbed/SC_Dataset/"
transform = transforms.Compose([
    transforms.Resize((299, 299)) if args.model_name in ['inceptionv3', 'qinceptionv3'] else transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(299) if args.model_name in ['inceptionv3', 'qinceptionv3'] else transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
batch_size = args.batch_size
num_epochs = 100
train_snr = 15
noise_type = args.noise_type

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
logdir = f"logs/log-{args.dataset}-{args.model_name}-{timestamp}/" if args.load_model is None else args.load_model
model_save_path = f'{logdir}/final_model.pth'


dataset = datasets.ImageFolder(root=root_folder, transform=transform)
subset_indices = torch.randperm(len(dataset)).tolist()
subset = Subset(dataset, subset_indices)
train_size = int(0.7 * len(subset))
train_dataset, test_dataset = torch.utils.data.random_split(subset, [train_size, len(subset) - train_size])

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(dataset.classes)

# Model selection
if args.model_name == 'inceptionv3':
    model = InceptionV3(num_classes, device).to(device)
elif args.model_name == 'qinceptionv3':
    model = QInceptionV3(num_classes, device).to(device)
elif args.model_name == 'resnet':
    model = ResNet50(num_classes, device).to(device)
elif args.model_name == 'qresnet':
    model = QResNet50(num_classes, device).to(device)
elif args.model_name == 'vit':
    model = ViT32(num_classes, device).to(device)
elif args.model_name == 'qvit':
    model = QViT32(num_classes, device).to(device)
elif args.model_name == 'swin':
    model = SwinT(num_classes, device).to(device)
elif args.model_name == 'qswin':
    model = QSwinT(num_classes, device).to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir=logdir)

def save_confusion_matrix_csv(cm, epoch):
    """ Save the confusion matrix to a CSV file. """
    cm_df = pd.DataFrame(cm)
    full_filename = f"{logdir}/cm_epoch{epoch}.csv"
    cm_df.to_csv(full_filename, index=False)

def train(model, dataloader):
    model.train()  # Set the model to training mode
    train_loss = 0
    predictions = []
    ground_truths = []
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, train_snr, noise_type)
        loss = criterion(outputs, labels) if not isinstance(outputs, tuple) else \
               criterion(outputs[0], labels) + 0.4 * criterion(outputs[1], labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs[0].data if isinstance(outputs, tuple) else outputs.data, 1)
        predictions.extend(preds.cpu().numpy())
        ground_truths.extend(labels.cpu().numpy())

    train_loss /= len(dataloader)
    accuracy = accuracy_score(ground_truths, predictions)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {accuracy:.4f}%")
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)


def test(model, dataloader, test_snr=train_snr, is_psnr=False):
    # model.eval()  # Set the model to evaluation mode
    test_loss = 0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, test_snr, noise_type)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    test_loss /= len(dataloader)
    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, average='macro')
    precision = precision_score(ground_truths, predictions, average='macro')
    recall = recall_score(ground_truths, predictions, average='macro')
    cm = confusion_matrix(ground_truths, predictions)
    save_confusion_matrix_csv(cm, epoch) 
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    if not is_psnr:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        writer.add_scalar('F1/test', f1, epoch)
        writer.add_scalar('Precision/test', precision, epoch)
        writer.add_scalar('Recall/test', recall, epoch)

# In case load model
if args.load_model is not None:
    model.load_state_dict(torch.load(model_save_path))

for epoch in range(num_epochs):
    train(model, dataloader_train)
    if epoch % 20 == 0:
        test(model, dataloader_test, 100)
        torch.save(model.state_dict(), model_save_path)


# PSNR Test for SNR 1-30
snr_values = list(range(1, 31))
for i in snr_values:
    test(model, dataloader_test, test_snr=i)
