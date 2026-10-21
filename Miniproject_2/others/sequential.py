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
