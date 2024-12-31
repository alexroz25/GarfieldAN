import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Hyperparameters
latent_dim = 50
img_size = 64
channels = 3
batch_size = 16  # Reduced for small dataset
lr_g = 0.0003
lr_d = 0.0003
epochs = 400

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # this normalizes img pixel values into [-1, 1]
])

dataset = datasets.ImageFolder(root="processed_datasets/garfield_cropped", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )

        self.adv_layer = nn.Sequential(nn.Linear(512 * (img_size // 16) ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity

generator = Generator()
discriminator = Discriminator()

# Loss and optimizers
adversarial_loss = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

# historic loss values for graphing
g_losses = []
d_losses = []

# Training Loop
for epoch in range(epochs+1):
    for i, (imgs, _) in enumerate(dataloader):

        # Real images
        real_imgs = imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(real_imgs.device)
        fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(real_imgs.device)

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn((imgs.size(0), latent_dim)).to(real_imgs.device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        g_optimizer.step()

        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if i == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    # show images
    if epoch % 400 == 0:
        with torch.no_grad():
            sample_z = torch.randn(25, latent_dim).to(real_imgs.device)
            generated = generator(sample_z).cpu()
            grid = torchvision.utils.make_grid(generated, nrow=5, normalize=True)
            plt.imshow(grid.permute(1, 2, 0).detach().numpy())
            plt.show()

        # plot loss graphs
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label="Generator Loss")
        plt.plot(d_losses, label="Discriminator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Generator and Discriminator Loss During Training")
        plt.show()

# Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
