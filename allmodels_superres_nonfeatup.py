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
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 512 * 7 * 7),  # Adjusting to match the starting feature map size
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 112x112 -> 224x224
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)  # Reshape to (batch_size, 512, 7, 7)
        x = self.deconv(x)
        return x
    

class DecoderInception(nn.Module):
    def __init__(self):
        super(DecoderInception, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 512 * 8 * 8),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128 -> 256x256
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),              # Adjust to final shape
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)  # Reshape to (batch_size, 512, 8, 8)
        x = self.deconv(x)
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return x



class InceptionV3(nn.Module):
    def __init__(self, num_classes, device):
        super(InceptionV3, self).__init__()
        self.encoder = InceptionEncoder().to(device)
        self.decoder = DecoderInception().to(device)
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
        self.decoder = DecoderInception().to(device) 
        self.flatten = nn.Flatten()
        self.final_layer = nn.Linear(n_qubits, num_classes)
        self.qnet = QNet().to(device)
        self.before_layer = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, n_qubits)
        )
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
        x2 = self.before_layer(x)
        x2 = self.qnet(x2)
        x2 = self.final_layer(x2)
        x = self.decoder(x + x2)
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
        self.decoder = Decoder().to(device)
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
        self.decoder = Decoder().to(device)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.qnet = QNet().to(device)
        self.before_layer = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, n_qubits)
        )
        self.final_layer = nn.Linear(n_qubits, num_classes)

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x2 = self.before_layer(x)
        x2 = self.qnet(x2)
        x2 = self.final_layer(x2)
        x = self.decoder(x + x2)
        return x


# Vision Transformer
class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.vit = ViT(
            image_size = 224,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    def forward(self, x):
        return self.vit(x)

class ViT32(nn.Module):
    def __init__(self, num_classes, device):
        super(ViT32, self).__init__()
        self.encoder = ViTEncoder().to(device)
        self.decoder = Decoder().to(device)
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
        self.decoder = Decoder().to(device) 
        self.flatten = nn.Flatten()
        self.qnet = QNet().to(device)
        self.final_layer = nn.Linear(n_qubits, num_classes)
        self.relu = nn.ReLU()
        self.before_layer = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, n_qubits)
        )

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x2 = self.before_layer(x)
        x2 = self.qnet(x2)
        x2 = self.final_layer(x2)
        x = self.decoder(x + x2)
        return x


# Swin Transformer
class SwinEncoder(nn.Module):
    def __init__(self, device):
        super(SwinEncoder, self).__init__()
        self.swin = SwinTransformer(
            in_chans=3,
            window_size=7,
            num_classes=1000,
            drop_rate=0.1,
        ).to(device)
    def forward(self, x):
        return self.swin(x)

class SwinT(nn.Module):
    def __init__(self, num_classes, device):
        super(SwinT, self).__init__()
        self.encoder = SwinEncoder(device).to(device) 
        self.decoder = Decoder().to(device) 
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
        self.decoder = Decoder().to(device) 
        self.flatten = nn.Flatten()
        self.qnet = QNet().to(device)
        self.final_layer = nn.Linear(n_qubits, num_classes)
        self.relu = nn.ReLU()
        self.before_layer = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, n_qubits)
        )

    def forward(self, x, snr=None, noise_type=None):
        x = self.encoder(x)
        x = self.flatten(x) 
        if noise_type == 'awgn' and snr is not None:
            x = add_awgn_noise(x, snr)
        elif noise_type == 'rayleigh' and snr is not None:
            x = add_rayleigh_noise(x, snr)
        x2 = self.before_layer(x)
        x2 = self.qnet(x2)
        x2 = self.final_layer(x2)
        x = self.decoder(x + x2)
        return x

    
