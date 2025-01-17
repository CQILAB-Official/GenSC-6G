import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import pennylane as qml
import argparse
from vit_pytorch import ViT
from architecture.swin_transformer import SwinTransformer
from semantic_communication.noise import add_awgn_noise, add_rayleigh_noise

# Quantum node setup
n_qubits = 10
dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

weight_shapes = {"weights": (10, n_qubits)}
# expanded_circuit = qml.transforms.broadcast_expand(qnode)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    def forward(self, x):
        return self.qlayer(x)


class InceptionEncoder(nn.Module):
    def __init__(self):
        super(InceptionEncoder, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)

    def forward(self, x):
        if self.training:
            x, aux = self.inception(x)
            return x, aux
        else:
            x = self.inception(x)
            return x

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes, device):
        super(InceptionV3, self).__init__()
        self.encoder = InceptionEncoder().to(device)
        self.decoder = Decoder(num_classes=num_classes).to(device)
        self.flatten = nn.Flatten()

    def forward(self, x, snr=None, noise_type=None):
        if self.training:
            x, aux = self.encoder(x)
        else:
            x = self.encoder(x)
        x = self.flatten(x) 
        # Adding noise to the encoded features
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x = self.decoder(x)
        return x

class QInceptionV3(nn.Module):
    def __init__(self, num_classes, device):
        super(QInceptionV3, self).__init__()
        self.encoder = InceptionEncoder().to(device)
        self.decoder = Decoder(num_classes=n_qubits).to(device) 
        self.flatten = nn.Flatten()
        self.final_layer = nn.Linear(n_qubits, num_classes)
        self.qnet = QNet().to(device)
        self.relu = nn.ReLU()

    def forward(self, x, snr=None, noise_type=None):
        if self.training:
            x, aux = self.encoder(x)
        else:
            x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x1 = self.decoder(x)
        x = self.qnet(x1)
        x = self.relu(x)
        x = x + x1
        x = self.final_layer(x)
        return x



# ResNet-50
class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes, device):
        super(ResNet50, self).__init__()
        self.encoder = ResNetEncoder().to(device)
        self.decoder = Decoder(num_classes=num_classes).to(device)
        self.flatten = nn.Flatten()

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x = self.decoder(x)
        return x

class QResNet50(nn.Module):
    def __init__(self, num_classes, device):
        super(QResNet50, self).__init__()
        self.encoder = ResNetEncoder().to(device)
        self.decoder = Decoder(num_classes=n_qubits).to(device)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.qnet = QNet().to(device)
        self.final_layer = nn.Linear(n_qubits, num_classes)

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x1 = self.decoder(x)
        x = self.qnet(x1)
        x = self.relu(x)
        x = x + x1
        x = self.final_layer(x)
        return x


# Vision Transformer
class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.vit = models.vit_l_32(pretrained=True)
        # ViT(
        #     image_size = 224,
        #     patch_size = 32,
        #     num_classes = 1000,
        #     dim = 1024,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
    def forward(self, x):
        return self.vit(x)

class ViT32(nn.Module):
    def __init__(self, num_classes, device):
        super(ViT32, self).__init__()
        self.encoder = ViTEncoder().to(device)
        self.decoder = Decoder(num_classes=num_classes).to(device)
        self.flatten = nn.Flatten()

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x = self.decoder(x)
        return x
        
class QViT32(nn.Module):
    def __init__(self, num_classes, device):
        super(QViT32, self).__init__()
        self.encoder = ViTEncoder().to(device)
        self.decoder = Decoder(num_classes=n_qubits).to(device) 
        self.flatten = nn.Flatten()
        self.qnet = QNet().to(device)
        self.final_layer = nn.Linear(n_qubits, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x1 = self.decoder(x)
        x = self.qnet(x1)
        x = self.relu(x)
        x = x + x1
        x = self.final_layer(x)
        return x


# Swin Transformer
class SwinEncoder(nn.Module):
    def __init__(self):
        super(SwinEncoder, self).__init__()
        self.swin = SwinTransformer(
            in_chans=3,
            window_size=7,
            num_classes=1000,
            dropout_rate=0.1,
            
        )
    def forward(self, x):
        return self.swin(x)

class SwinT(nn.Module):
    def __init__(self, num_classes, device):
        super(SwinT, self).__init__()
        self.encoder = SwinEncoder(device).to(device) 
        self.decoder = Decoder(num_classes=num_classes).to(device) 
        self.flatten = nn.Flatten()

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x = self.decoder(x)
        return x

class QSwinT(nn.Module):
    def __init__(self, num_classes, device):
        super(QSwinT, self).__init__()
        self.encoder = SwinEncoder(device).to(device) 
        self.decoder = Decoder(num_classes=n_qubits).to(device) 
        self.flatten = nn.Flatten()
        self.qnet = QNet().to(device)
        self.final_layer = nn.Linear(n_qubits, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x1 = self.decoder(x)
        x = self.qnet(x1)
        x = self.relu(x)
        x = x + x1
        x = self.final_layer(x)
        return x

# MobileNet
class MobileNetEncoder(nn.Module):
    def __init__(self):
        super(MobileNetEncoder, self).__init__()
        self.mnet = models.mobilenet_v3_large(pretrained=True)

    def forward(self, x):
        x = self.mnet(x)
        return x
    
# EfficientNet
class EfficientNetEncoder(nn.Module):
    def __init__(self):
        super(EfficientNetEncoder, self).__init__()
        self.enet = models.efficientnet_b1(pretrained=True)

    def forward(self, x):
        x = self.enet(x)
        return x
    
# VGG
class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.vgg = models.vgg16(pretrained=True)

    def forward(self, x):
        x = self.vgg(x)
        return x
    
# U-Net
class QCNNEncoder(nn.Module):
    def __init__(self):
        super(QCNNEncoder, self).__init__()
        self.qcnn = models.quantization.resnet18(pretrained=True)

    def forward(self, x):
        x = self.qcnn(x)
        return x


# class QuantumInceptionV3(nn.Module):
#     def __init__(self, num_classes):
#         super(QuantumInceptionV3, self).__init__()
#         self.inception = models.inception_v3(pretrained=True, aux_logits=True)
#         for param in self.inception.parameters():
#             param.requires_grad = False
#         self.inception.fc = nn.Linear(self.inception.fc.in_features, n_qubits)
#         self.inception.AuxLogits.fc = nn.Linear(self.inception.AuxLogits.fc.in_features, n_qubits)
#         self.qnet = QNet().to(device)
#         self.fc_final = nn.Linear(n_qubits, num_classes)
#     def forward(self, x, snr=None, noise_type=None):
#         if self.training:
#             x, aux = self.inception(x)
#             x = self.qnet(x)
#             aux = self.qnet(aux)
#             x = self.fc_final(x)
#             aux = self.fc_final(aux)
#             return x, aux
#         else:
#             x = self.inception(x)
#             x = self.qnet(x)
#             x = self.fc_final(x)
#             return x

class CNNEncoder(nn.Module):
    def __init__(self, num_classes=1000):
        super(CNNEncoder, self).__init__()
        
        self.features = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 7x7 is the output size after the conv layers and max-pooling
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  # Output 1000 features
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = self.classifier(x)
        return x

