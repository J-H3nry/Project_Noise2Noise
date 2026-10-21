from pathlib import Path

import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import time


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR.parent.parent

train_data_path = DATA_DIR / 'train_data.pkl'
val_data_path = DATA_DIR / 'val_data.pkl'

sys.path.insert(0, str(MODEL_DIR))
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training = False #if we want to train the model, set it to True, else False

# load data
train_input, train_target = torch.load(train_data_path)  # noisy images 1, noisy images 2
val_input, val_target = torch.load(val_data_path)  # noisy images, clean images

# convert from ByteTensor to FloatTensor
train_input = train_input.to(device)
train_target = train_target.to(device)

val_input = val_input.to(device)
val_target = val_target

# train
model = Model()

if training:
    epoch = 100
    start = time.perf_counter()
    model.train(train_input, train_target, epoch)
    stop = time.perf_counter()
    model.save_model()
    print(f'The training is done in {stop - start} seconds')


model.load_pretrained_model()

prediction = model.predict(val_input)

# Show results
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

psnr = model.psnr(val_target)
print(f'psnr is {psnr} dB')
print(f'The training is done in {stop - start} seconds')