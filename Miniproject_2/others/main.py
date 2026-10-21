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

torch.set_grad_enabled(False)

gridsearch = True #if we want to do a gridsearch for the hyperparameters, set it to True, else False
training = False #if we want to train the model, set it to True, else False

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
epoch = 100

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
print(f'psnr is {psnr_value} dB')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

images = [
    val_input[image_so_see],       # noisy
    prediction[image_so_see],      # denoised
    val_target[image_so_see],      # clean
]
titles = ["Noisy", "Denoised", "Clean"]

for ax, img, title in zip(axes, images, titles):
    img = img.detach().cpu().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.show()
