import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Definir la estructura del modelo
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 10)  # 10 clases para MNIST

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Cambiar a (num_tokens, batch_size, input_dim) para Transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Promedio sobre la dimensión de los tokens
        x = self.fc(x)
        return x

# Hiperparámetros (deben coincidir con los usados para entrenar el modelo)
input_dim = 28
num_heads = 4
ff_dim = 128
num_layers = 2

# Inicializar el modelo
model = TransformerEncoder(input_dim, num_heads, ff_dim, num_layers)

# Cargar los pesos del archivo .pth
model.load_state_dict(torch.load('transformer_model.pth'))

# Poner el modelo en modo de evaluación
model.eval()

# Preparar el dataset de prueba
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Función para evaluar el modelo
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28, 28)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# Evaluar el modelo
test(model, test_loader)
