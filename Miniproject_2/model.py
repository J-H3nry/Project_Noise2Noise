
# For mini-project 2

from torch import device, empty, set_grad_enabled, Tensor
import torch
from torch.nn.functional import fold, unfold
from torch.nn import init
from pathlib import Path
import pickle

set_grad_enabled(False)

class Model():
    """Denoising convolutional neural network and training interface.

    The model maps noisy RGB images to denoised images using a small
    convolutional encoder-decoder network.
    """

    def __init__(self) -> None:
        """Initialize the network, optimizer, and mean squared error loss."""

        self.device = device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_device(self.device)

        self.model = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=2),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, stride=2),
            ReLU(),
            TransposeConv2d(32, 32, kernel_size=3, stride=2),
            ReLU(),
            TransposeConv2d(32, 3, kernel_size=3, stride=2, output_padding= 1),
            Sigmoid()
        )
        self.model_parameters = self.model.param()

        self.optimizer = SGD(model_parameters= self.model_parameters, lr= 0.01)
        self.loss = MSE()
        self.output = None

    def load_pretrained_model(self) -> None:
        """Load the pretrained parameters from ``bestmodel2.pth``.

        The checkpoint is expected to be stored next to this module.
        """
        model_path = Path(__file__).parent / "bestmodel2.pth"
        i = 0
        with open(model_path, 'rb') as handle:
            bestmodel =pickle.load(handle)
        for parameters in self.model_parameters:
            if parameters:
                parameters[0][0].copy_(bestmodel[i]['weight'])
                parameters[1][0].copy_(bestmodel[i]['bias'])
                i += 1
    def save_model(self):
        """Save the network parameters to ``bestmodel2.pth``.

        The checkpoint is written next to this module.
        """
        model_path = Path(__file__).parent/"bestmodel2.pth"
        model_param = {}
        layer_number = 0
        for parameters in self.model_parameters:
            if parameters:
                params = {}
                params['weight'] = parameters[0][0]
                params['bias'] = parameters[1][0]
                model_param[layer_number] = params
                layer_number += 1
        with open(model_path, 'wb') as handle:
            pickle.dump(model_param, handle)


    def train(self, train_input, train_target, num_epochs, learning_rate=None) -> None:
        """Train the denoising network on paired noisy images.

        Args:
            train_input (Tensor): Noisy input images of shape ``(N, C, H, W)``.
            train_target (Tensor): Target images of shape ``(N, C, H, W)``.
            num_epochs (int): Number of training epochs.
            learning_rate (float, optional): Learning rate for this step. If not provided, the learning rate specified during initialization is used.

        Inputs are converted to floating point and scaled to the ``[0, 1]``
        range before training.
        """
        set_grad_enabled(False)
        lr = learning_rate if learning_rate is not None else self.optimizer.lr

        batch_size = 50
        train_input = train_input.float().to(self.device)/255
        train_target = train_target.float().to(self.device)/255

        for epoch in range(num_epochs):

            for b in range(0, train_input.size(0), batch_size):
                output = self.model(train_input.narrow(0, b, batch_size))
                loss = self.loss(output, train_target.narrow(0, b, batch_size))
                grad = self.loss.backward()
                self.model.backward(grad)
                self.optimizer.step(lr=lr)

            print(f'epoch {epoch}: loss is {loss}')

    def predict(self, test_input):
        """Denoise a batch of images.

        Args:
            test_input (Tensor): Input images of shape ``(N, C, H, W)``.

        Returns:
            Tensor: Denoised images with the same shape as ``test_input``,
            returned on the CPU and scaled to the ``[0, 255]`` range.
        """
        test_input = test_input.float().to(self.device)/255

        self.output = self.model(test_input)
        
        self.output = self.output.cpu() * 255
        return self.output

##############################################################################################################################################


class MSE():
    """
    Mean squared error loss (reduction='mean' only).

    Both inputs must be tensors of identical shape, dtype and device.
    """

    def __init__(self):
        self.mask = None

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the mean squared error between the prediction and the ground truth.

        Args:
            x (Tensor): Prediction returned by the model.
            y (Tensor): Ground truth tensor.

        Returns:
            Tensor: The computed MSE loss.
        """
        return self.forward(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        output = pow(x - y, 2).mean()
        self.mask = (x-y).mul(2)/x.numel()
        return output

    def backward(self):
        """
        Computes the gradient of the MSE loss with respect to the input tensor.

        Returns:
            Tensor: The computed gradient of the MSE loss.
        Raises:
            ValueError: If the `forward` method has not been called before calling `backward`.
        """
        if self.mask is None:
            raise ValueError("Forward method must be called before backward.")
        
        return self.mask


class ReLU():
    """
    Apply the rectified linear unit (ReLU) activation element-wise.

    Negative input values are set to zero, while nonnegative values are
    retained. The input ``x`` must be a tensor. The layer stores a mask during
    the forward pass so ``backward`` can propagate gradients only where the
    input was positive.
    """
    def __init__(self):
        self.mask = None

    def __call__(self, x):
        """
        Apply the ReLU activation to the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The input with negative values set to zero.
        """
        return self.forward(x)

    def forward(self, x):
        output = x.clone()
        output[output < 0] = 0
        output[output > 0] = 1
        self.mask = output
        output = output.mul(x)
        return output

    def backward(self, dx):
        """
        Backpropagate through the ReLU activation.

        Args:
            dx (Tensor): Gradient of the loss with respect to the output.

        Raises:
            ValueError: If `forward` has not been called first.

        Returns:
            Tensor: Gradient of the loss with respect to the input.
        """
        if self.mask is None:
            raise ValueError("Forward method must be called before backward.")
        output = self.mask * dx
        return output

    def param(self):
        """
        Return the parameters of this activation.

        Returns:
            list: Empty, because ReLU has no trainable parameters.
        """
        return []


class Sigmoid():
    """
    Apply the sigmoid activation element-wise.
    """

    def __init__(self):
        self.mask = None

    def __call__(self, x):
        """
        Apply sigmoid to ``x`` and cache its derivative.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The element-wise sigmoid of ``x``.
        """
        return self.forward(x)

    def forward(self, x):
        """Compute sigmoid values and cache the derivative for backpropagation."""
        output = 1 / (1 + x.mul(-1).exp())
        self.mask = x.mul(-1).exp()/(1 + x.mul(-1).exp())**2
        return output

    def backward(self, dx):
        """
        Backpropagate through sigmoid using the derivative cached by the last call.

        Args:
            dx (Tensor): Gradient of the loss with respect to the output.

        Raises:
            ValueError: If the activation has not been called first.

        Returns:
            Tensor: Gradient of the loss with respect to the input.
        """
        if self.mask is None:
            raise ValueError("Forward method must be called before backward.")
        output = self.mask*dx
        return output

    def param(self):
        """
        Return the parameters of this activation.

        Returns:
            list: Empty, because sigmoid has no trainable parameters.
        """
        return []


class Conv2d():
    """
    Apply a two-dimensional convolution to a batch of input tensors.

    The input is expected to have shape ``(batch, in_channels, height, width)``.
    Parameters and their gradients are stored on the instance.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple, optional): Height and width of the kernel.
        stride (int or tuple, optional): Convolution step along each axis.
        padding (int or tuple, optional): Implicit padding on each side.
        dilation (int or tuple, optional): Spacing between kernel elements.
        bias (bool, optional): Whether to use a bias term. Currently a bias
            vector is initialized regardless of this value.
    """

    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = True):
        """
        Apply a two-dimensional convolution to a batch of input tensors.

        The input is expected to have shape ``(batch, in_channels, height, width)``.
        Parameters and their gradients are stored on the instance.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple, optional): Height and width of the kernel.
            stride (int or tuple, optional): Convolution step along each axis.
            padding (int or tuple, optional): Implicit padding on each side.
            dilation (int or tuple, optional): Spacing between kernel elements.
            bias (bool, optional): Whether to use a bias term. Currently a bias
                vector is initialized regardless of this value.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.weight = init.kaiming_normal_(empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = init.zeros_(empty(self.out_channels))
        self.dW = empty(self.weight.shape)
        self.db = empty(self.bias.shape)

    def __call__(self, x):
        """
        Compute the convolution and cache input data for backpropagation.

        Args:
            x (Tensor): Input with shape ``(batch, in_channels, height, width)``.

        Returns:
            Tensor: Convolution output.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the convolution and cache input data for backpropagation.

        Args:
            x (Tensor): Input with shape ``(batch, in_channels, height, width)``.

        Returns:
            Tensor: Output with shape ``(batch, out_channels, out_height, out_width)``.
        """
        self.x = x
        batch_size = x.size(0)
        self.output_size_h = int((x.shape[2] - self.dilation[0] * (self.kernel_size[0] - 1) - 1 + 2 * self.padding[0])/self.stride[0]) + 1
        self.output_size_w = int((x.shape[3] - self.dilation[1] * (self.kernel_size[1] - 1) - 1 + 2 * self.padding[1])/self.stride[1]) + 1
        self.unfolded = unfold(x, kernel_size= self.kernel_size, padding= self.padding, stride= self.stride, dilation= self.dilation)
        weight = self.weight.view(self.out_channels, -1)
        wxb = weight @ self.unfolded + self.bias.view(1, -1, 1)
        result = wxb.view(batch_size, self.out_channels, self.output_size_h, self.output_size_w)
        return result

    def set_weight(self, weight):
        """
        Replace the convolution kernel values in place.

        Args:
            weight (Tensor): New weights, with the same shape as ``self.weight``.
        """
        self.weight *= 0
        self.weight += weight

    def set_bias(self, bias):
        """
        Replace the bias values in place.

        Args:
            bias (Tensor): New bias values, with the same shape as ``self.bias``.
        """
        self.bias *= 0
        self.bias += bias

    def backward(self, output_grad):
        """
        Backpropagate output gradients and store parameter gradients.

        The forward method must be called before this method.

        Args:
            output_grad (Tensor): Gradient with respect to the output, shaped
                ``(batch, out_channels, out_height, out_width)``.

        Returns:
            Tensor: Gradient with respect to the input.
        """
        batch_size = output_grad.size(0)
        self.db *= 0
        self.db += output_grad.sum((0, 2, 3))
        da = output_grad.permute(1, 2, 3, 0).reshape(self.out_channels, -1)
        unfolded_reshaped = self.unfolded.permute(1, 2, 0).reshape(self.unfolded.shape[1], -1)
        self.dW *= 0
        self.dW += (da @ unfolded_reshaped.T).reshape(self.weight.shape)

        weight = self.weight.view(self.out_channels, -1)
        L = self.output_size_h * self.output_size_w
        input_grad_reshaped = (weight.T @ da).view(self.x.shape[1] * self.kernel_size[0] * self.kernel_size[1], L, batch_size).permute(2, 0, 1)
        input_grad = fold(input_grad_reshaped, output_size= (self.x.shape[2:]), kernel_size= self.kernel_size, padding= self.padding, dilation= self.dilation, stride=self.stride)
        return input_grad

    def param(self):
        """
        Return trainable parameters paired with their gradient buffers.

        Returns:
            list: ``[[weight, weight_gradient], [bias, bias_gradient]]``.
        """
        return [[self.weight, self.dW], [self.bias, self.db]]


class TransposeConv2d():
    """
    Apply a 2D transposed convolution to an input feature map.

    Input shape:
        (N, in_channels, H_in, W_in), where N is the batch size.

    Output shape:
        (N, out_channels, H_out, W_out), where each spatial dimension is
        computed as:
        (input_size - 1) * stride - 2 * padding
        + dilation * (kernel_size - 1) + output_padding + 1.
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the layer.
        kernel_size (tuple or int, optional): Size of the convolution kernel.
        stride (tuple or int, optional): Stride of the convolution.
        padding (tuple or int, optional): Padding added to the input.
        output_padding (tuple or int, optional): Additional size added to the output.
        dilation (tuple or int, optional): Spacing between kernel elements.
        bias (bool, optional): If True, a bias vector is added to the output. Currently, a bias vector is initialized regardless of this value.
    """

    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), stride = 1, padding = 0, output_padding = 0, dilation = 1, bias = True):
        """
        Apply a 2D transposed convolution to an input feature map.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the layer.
            kernel_size (tuple or int, optional): Size of the convolution kernel.
            stride (tuple or int, optional): Stride of the convolution.
            padding (tuple or int, optional): Padding added to the input.
            output_padding (tuple or int, optional): Additional size added to the output.
            dilation (tuple or int, optional): Spacing between kernel elements.
            bias (bool, optional): If True, a bias vector is added to the output. Currently, a bias vector is initialized regardless of this value.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.weight = init.kaiming_normal_(empty(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = init.zeros_(empty(self.out_channels))
        self.dW = empty(self.weight.shape)
        self.db = empty(self.bias.shape)
        

    def __call__(self, x):
        """
        Compute the forward pass of the transposed convolution.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, H_out, W_out).
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass of the transposed convolution.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, H_out, W_out).
        """
        self.x = x
        batch_size = x.size(0)
        H = x.shape[2]
        W = x.shape[3]
        self.output_size_h = (H - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        self.output_size_w = (W - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        weight = self.weight.view(self.in_channels, -1)
        self.x_reshaped = x.permute(1, 2, 3, 0).reshape(self.in_channels, -1)

        output_reshaped = (weight.T @ self.x_reshaped).view(self.out_channels * self.kernel_size[0] * self.kernel_size[1], -1, batch_size).permute(2, 0, 1)
        output = fold(output_reshaped, output_size=(self.output_size_h, self.output_size_w), kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride)
        output = output + self.bias.view(1, -1, 1, 1)
        return output


    def backward(self, output_grad):
        """
        Compute gradients of the loss with respect to the input and parameters.

        Args:
            output_grad (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tensor: Gradient of the loss with respect to the input tensor.
        """
        batch_size = self.x.size(0)
        self.db *= 0
        self.db += output_grad.sum((0, 2, 3)).view(-1)
        grad_reshaped = unfold(output_grad, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride,dilation=self.dilation).permute(1, 0, 2)
        grad_reshaped_out = grad_reshaped.reshape(grad_reshaped.shape[0], -1)
        weight = self.weight.permute(1, 2, 3, 0).reshape(self.out_channels * self.kernel_size[0] * self.kernel_size[1], -1)
        output = (weight.T @ grad_reshaped_out).view(self.in_channels, batch_size, self.x.shape[2], self.x.shape[3]).permute(1, 0, 2, 3)
        grad_reshaped_w = grad_reshaped.permute(2, 1, 0)
        grad_reshaped_w = grad_reshaped_w.reshape(self.x_reshaped.shape[1], -1)
        self.dW *= 0
        self.dW += (self.x_reshaped @ grad_reshaped_w).view(self.weight.shape)

        return output

    def param(self):
        """
        Return trainable parameters paired with their gradient buffers.

        Returns:
            list: ``[[weight, weight_gradient], [bias, bias_gradient]]``.
        """
        return [[self.weight, self.dW], [self.bias, self.db]]


class SGD():
    """
    Implements stochastic gradient descent.

    Updates each parameter in-place using its gradient and the learning rate.

    Args:
        model_parameters (iterable): Groups of ``(parameter, gradient)`` pairs.
        lr (float): Learning rate used for each update.
    """
    def __init__(self, model_parameters, lr):
        """
        Initialize the optimizer.

        Args:
            model_parameters (iterable): Groups of ``(parameter, gradient)`` pairs.
            lr (float): Learning rate used for each update.
        """
        self.model_parameters = model_parameters
        self.lr = lr

    def zero_grad(self):
        """Reset the stored gradients to zero."""
        for parameters in self.model_parameters:
            for element in parameters:
                element[0][1] *= 0
                element[1][1] *= 0

    def step(self, lr=None):
        """
        Perform a single optimization step.

        Each parameter is updated in-place according to
        ``parameter -= lr * gradient``.

        Args:
            lr (float, optional): Learning rate for this step. If not provided, the learning rate specified during initialization is used.
        """
        if lr is not None:
            self.lr = lr
        
        for parameters in self.model_parameters:
            for element in parameters:
                element[0] -= self.lr * element[1]


class Sequential():
    """A sequential container for applying layers in order.

    Layers are called from first to last during the forward pass and in
    reverse order during the backward pass.

    Args:
        *args: Layers to apply, each implementing ``forward``, ``backward``,
            and ``param`` methods.
    """
    def __init__(self, *args):
        self.args = args
        self.parameters = []

    def __call__(self, in_data):
        """Run the forward pass on the input.

        Args:
            in_data: Input passed to the first layer.

        Returns:
            The output produced by the final layer.
        """
        return self.forward(in_data)

    def forward(self, in_data):
        """Apply each layer in sequence.

        Args:
            in_data: Input passed to the first layer.

        Returns:
            The output produced by the final layer.
        """
        for element in range(len(self.args)):
            in_data = self.args[element].forward(in_data)
        out_data = in_data

        return out_data

    def backward(self, in_data):
        """Propagate the supplied gradient through the layers in reverse.

        Args:
            in_data: Gradient or value passed to the final layer's backward
                method.

        Returns:
            The result returned by the first layer's backward method.
        """
        for element in reversed(range(len(self.args))):
            in_data = self.args[element].backward(in_data)
        out_data = in_data
        return out_data

    def param(self):
        """Collect parameter and gradient pairs from every layer.

        Returns:
            list: Parameter data returned by each layer's ``param`` method.
        """
        self.parameters = []
        for element in range(len(self.args)):
            self.parameters.append(self.args[element].param())  # [[self.weights, self.dW], [self.bias, self.db]]
        return self.parameters
