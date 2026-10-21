from pathlib import Path
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR.parent.parent

train_data_path = DATA_DIR / 'train_data.pkl'
val_data_path = DATA_DIR / 'val_data.pkl'


sys.path.insert(0, str(MODEL_DIR))

from model import Model

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    parser.add_argument('--train', action='store_true', help="train the model (default: only load the savec model and evaluate)")
    parser.add_argument('--gridsearch', action='store_true', help="do a gridsearch for the learning rate (default: False, requires --train to be set)")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs for training (default: 100)")
    return parser.parse_args()


args = parse_args()
training = args.train
gridsearch = args.gridsearch
epoch = args.epochs

#psnr
def psnr(x, y, max_range=1.0):
    x = x/255
    y = y/255
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()

# load data
train_input, train_target = torch.load(train_data_path)  # noisy images 1, noisy images 2
val_input, val_target = torch.load(val_data_path)  # noisy images, clean images
#


# train
if training:
    if gridsearch:
        best_psnr = 0
        learning_rates = np.logspace(-1, 1, num=8)
        for lr in learning_rates:
            model = Model()
            print(f'training with learning rate: {lr}')
            model.train(train_input, train_target, epoch, learning_rate=lr)
            prediction = model.predict(val_input)
            psnr_value = psnr(prediction, val_target)
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_lr = lr
                model.save_model()
        print(f'Best learning rate: {best_lr} with PSNR: {best_psnr} dB')
    else:
        model = Model()
        start = time.perf_counter()
        model.train(train_input, train_target, epoch)
        stop = time.perf_counter()
        model.save_model()
        prediction = model.predict(val_input)
        print(f'The training is done in {stop - start} seconds')
        psnr_value = psnr(prediction, val_target)
        print(f'psnr is {psnr_value} dB')

# Visualize the results
image_so_see = 1
model = Model()

model.load_pretrained_model()

prediction = model.predict(val_input)
psnr_value = psnr(prediction, val_target)
# print(f'psnr is {psnr_value} dB')


rows = 3
columns = 10
first_image = 100
fig, axes = plt.subplots(rows, columns, figsize=(16, 8))

count = first_image
for j in range(columns):
    axes[0, j].imshow(val_input[count].detach().cpu().clamp(0, 255).byte().permute(1, 2, 0).numpy())
    count = count + 1
axes[0][0].set_title('Test input')

count = first_image
for j in range(columns):
    axes[1, j].imshow(prediction[count].detach().cpu().clamp(0, 255).byte().permute(1, 2, 0).numpy())
    count = count + 1
axes[1][0].set_title('Prediction')

count = first_image
for j in range(columns):
    axes[2, j].imshow(val_target[count].detach().cpu().clamp(0, 255).byte().permute(1, 2, 0).numpy())
    count = count + 1
axes[2][0].set_title('Test target')

plt.show()
