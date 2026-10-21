from torch import empty
from torch.nn.functional import fold, unfold
from torch.nn import init

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
