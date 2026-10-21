import torch
from torch import nn
from pathlib import Path
# For mini-project 1

torch.set_grad_enabled(True)
class Model():
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Noise2Noise().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.005, betas= (0.9, 0.999), eps= 1e-08, weight_decay= 0.0, amsgrad= False)
        self.loss = nn.MSELoss()
        self.output = None

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        # pass

    def train(self, train_input, train_target, num_epochs) -> None:
        torch.set_grad_enabled(True)
        # :train_input: tensor of size (N, C, H, W) containing a noisy version of the images

        # :train_target: tensor of size (N, C, H, W) containing another noisy version of the
        # same images, which only differs from the input by their noise.
        batch_size = 50
        train_input = train_input.float().to(self.device)
        train_target = train_target.float().to(self.device)
        for epoch in range(num_epochs):

            for b in range(0, train_input.size(0), batch_size):
                output = self.model(train_input.narrow(0, b, batch_size))
                loss = self.loss(output, train_target.narrow(0, b, batch_size))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_model(self):
        torch.save(self.model.state_dict(), 'bestmodel.pth')

    def predict(self, test_input) -> torch.Tensor:
        # : test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained
        # or the loaded network.

        # : returns a tensor of the size (N1, C, H, W)
        test_input = test_input.float().to(self.device)
        self.output = self.model(test_input)
        self.output = torch.where(self.output <= 255, self.output, torch.tensor([255]).type(torch.FloatTensor).to(self.device))
        self.output = self.output.cpu()
        return self.output

    def psnr(self, ground_truth):
        x = self.output / 255
        y = ground_truth / 255
        return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


class Noise2Noise(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # encoder
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )

        # decoder
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(96),
            nn.Upsample(scale_factor=2)
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2)
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(3+64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # encoder                                           # Here are the ouput sizes of every sequential blocks of the network
        encoder1 = self.block1(x)                           # 32
        encoder2 = self.block2(encoder1)                    # 64
        encoder3 = self.block3(encoder2)                    # 128
        encoder4 = self.block4(encoder3)                    # 128

        # decoder
        decoder1 = self.block5(encoder4)                    # 128
        concat1 = torch.cat((decoder1, encoder3), dim=1)    # 128 + 128 = 256
        decoder2 = self.block6(concat1)
        concat2 = torch.cat((decoder2, encoder2), dim=1)    # 64 + 128  = 192
        decoder3 = self.block7(concat2)
        concat3 = torch.cat((decoder3, encoder1), dim=1)    # 32 + 96 = 128
        decoder4 = self.block8(concat3)
        concat4 = torch.cat((decoder4, x), dim=1)           # 3 + 64 = 67

        # Final activation
        return self.block9(concat4)


class Noise2NoisePaper_like_version(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # size of our images : 3x32x32 (3 channels(RGB) and height = width = 32)
        # formula to calculate convolution size : new_height = (h - K + 2*P)/S + 1

        # Encoder
        self.enc0 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(3, 3), padding='same'),  # ENC_CONV0
            nn.LeakyReLU()
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=(3, 3), padding='same'),  # ENC_CONV1
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))  # POOL1 -> 32/2 = 16
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=(3, 3), padding='same'),  # ENC_CONV2
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))  # POOL2 -> 16/2 = 8
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=(3, 3), padding='same'),  # ENC_CONV3
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))  # POOL3 -> 8/2 = 4
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=(3, 3), padding='same'),  # ENC_CONV4
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))  # POOL4 -> 4/2 = 2
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=(3, 3), padding='same'),  # ENC_CONV5
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))  # POOL5 -> 2/2 = 1
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=(3, 3), padding='same'),  # ENC_CONV6
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),  # UPSAMPLE5
        )

        # Decoder
        self.dec5a = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV5a
            nn.LeakyReLU()
        )
        self.dec5b = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV5b
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)  # UPSAMPLE4
        )
        self.dec4a = nn.Sequential(
            nn.Conv2d(144, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV4a
            nn.LeakyReLU()
        )
        self.dec4b = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV4b
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),  # UPSAMPLE3
        )
        self.dec3a = nn.Sequential(
            nn.Conv2d(144, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV3a
            nn.LeakyReLU()
        )
        self.dec3b = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV3b
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)  # UPSAMPLE2
        )
        self.dec2a = nn.Sequential(
            nn.Conv2d(144, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV2a
            nn.LeakyReLU()
        )
        self.dec2b = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding='same'),  # DEC_CONV2b
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),  # UPSAMPLE1
        )
        self.dec1a = nn.Sequential(
            nn.Conv2d(99, 64, kernel_size=(3, 3), padding='same'),  # DEC_CONV1a
            nn.LeakyReLU()
        )
        self.dec1b = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding='same'),  # DEC_CONV1b
            nn.LeakyReLU()
        )
        self.dec1c = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(3, 3), padding='same'),  # DEC_CONV1c
            nn.ReLU()
        )

    def forward(self, x):
        # encoder
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        concat5 = torch.cat((enc6, enc4), dim=1)  # 96

        # decoder
        dec5a = self.dec5a(concat5)
        dec5b = self.dec5b(dec5a)
        concat4 = torch.cat((dec5b, enc3), dim=1)  # 144
        dec4a = self.dec4a(concat4)
        dec4b = self.dec4b(dec4a)
        concat3 = torch.cat((dec4b, enc2), dim=1)  # 144
        dec3a = self.dec3a(concat3)
        dec3b = self.dec3b(dec3a)
        concat2 = torch.cat((dec3b, enc1), dim=1)  # 144
        dec2a = self.dec2a(concat2)
        dec2b = self.dec2b(dec2a)
        concat1 = torch.cat((dec2b, x), dim=1)  # 96 + 3 = 99
        dec1a = self.dec1a(concat1)
        dec1b = self.dec1b(dec1a)

        # Final activation
        return self.dec1c(dec1b)
