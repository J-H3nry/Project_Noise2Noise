from torch import empty, cat, arange

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