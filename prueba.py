import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Definir el Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 10)  # 10 clases para MNIST

    def forward(self, x):
        # Suponiendo que x es de forma (batch_size, num_tokens, input_dim)
        x = x.permute(1, 0, 2)  # Cambiar a (num_tokens, batch_size, input_dim) para Transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Promedio sobre la dimensión de los tokens
        x = self.fc(x)
        return x

# Hiperparámetros
input_dim = 28  # Dimensión de cada token (en este caso, la altura de la imagen de MNIST)
num_heads = 4
ff_dim = 128
num_layers = 2
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Preparar el dataset de MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Inicializar el modelo, la pérdida y el optimizador
model = TransformerEncoder(input_dim, num_heads, ff_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Función de entrenamiento
def train(model, train_loader, criterion, optimizer, num_epochs):
    """
    Función para entrenar el modelo.

    Args:
        model (nn.Module): El modelo a entrenar.
        train_loader (DataLoader): DataLoader que contiene los datos de entrenamiento.
        criterion (nn.Module): Función de pérdida.
        optimizer (torch.optim.Optimizer): Optimizador para actualizar los parámetros del modelo.
        num_epochs (int): Número de épocas de entrenamiento.

    Returns:
        None
    """
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28, 28)  # Redimensionar a (batch_size, num_tokens, input_dim)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Función de prueba
def test(model, test_loader, criterion):
    """
    Función para evaluar el modelo en el conjunto de prueba.

    Args:
        model (nn.Module): El modelo a evaluar.
        test_loader (DataLoader): DataLoader que contiene los datos de prueba.
        criterion (nn.Module): Función de pérdida.

    Returns:
        None
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28, 28)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')

# Entrenar y probar el modelo
train(model, train_loader, criterion, optimizer, num_epochs)
test(model, test_loader, criterion)
# Guardar el modelo entrenado
torch.save(model.state_dict(), 'transformer_model.pth')