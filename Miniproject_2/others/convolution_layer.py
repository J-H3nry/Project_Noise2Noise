from torch import empty
from torch.nn.functional import fold, unfold
from torch.nn import init


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
    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.weights = empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_()
        self.bias = empty(self.out_channels).uniform_()
        self.dW = empty(self.weights.shape)
        self.db = empty(self.bias.shape)

    def __call__(self, x):
        return self.forward(x)


    def forward(self, x):
        self.x = x
        batch_size = x.size(0)
        self.output_size_h = int((x.shape[2] - self.dilation[0] * (self.kernel_size[0] - 1) - 1 + 2 * self.padding[0])/self.stride[0]) + 1
        self.output_size_w = int((x.shape[3] - self.dilation[1] * (self.kernel_size[1] - 1) - 1 + 2 * self.padding[1])/self.stride[1]) + 1
        self.unfolded = unfold(x, kernel_size= self.kernel_size, padding= self.padding, stride= self.stride, dilation= self.dilation)
        weight = self.weights.view(self.out_channels, -1)
        wxb = weight @ self.unfolded + self.bias.view(1, -1, 1)
        result = wxb.view(batch_size, self.out_channels, self.output_size_h, self.output_size_w)
        return result

    def set_weight(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

    def backward(self, output_grad):
        batch_size = output_grad.size(0)
        self.db = output_grad.sum((0, 2, 3))
        da = output_grad.permute(1, 2, 3, 0).reshape(self.out_channels, -1)
        unfolded_reshaped = self.unfolded.permute(1, 2, 0).reshape(self.unfolded.shape[1], -1)

        self.dW = (da @ unfolded_reshaped.T).view(self.weights.shape)

        weight = self.weights.view(self.out_channels, -1)
        L = self.output_size_h * self.output_size_w
        input_grad_reshaped = (weight.T @ da).view(self.x.shape[1] * self.kernel_size[0] * self.kernel_size[1], L, batch_size).permute(2, 0, 1)
        input_grad = fold(input_grad_reshaped, output_size= (self.x.shape[2:]), kernel_size= self.kernel_size, padding= self.padding, dilation= self.dilation, stride=self.stride)
        return input_grad

    def param(self):
        return [[self.weights, self.dW], [self.bias, self.db]]
