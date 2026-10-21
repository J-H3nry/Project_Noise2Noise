import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import torchvision.transforms

sys.path.insert(1, '../')

train_input, train_target = torch.load('../../train_data.pkl')  # noisy images 1, noisy images 2
val_input, val_target = torch.load('../../val_data.pkl')  # noisy images, clean images

# normalize and convert from ByteTensor to FloatTensor
train_input = train_input.float().div(255)
train_target = train_target.float().div(255)

val_input = val_input.float().div(255)
val_target = val_target.float().div(255)

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(30),
    torchvision.transforms.ColorJitter(brightness=0.5, saturation=0.5)
])
train_input_augment = transform(train_input)
train_input_augmented = torch.cat((train_input, train_input_augment), dim = 0)
print(f'the size is {train_input_augmented.size()}')

# plt.close('all')
# for i in range(10):
#     image_test = val_target[i].mul(255).numpy().astype(np.uint8)
#     image_test = np.transpose(image_test, (1, 2, 0))
#     plt.imshow(image_test)
#     plt.show()