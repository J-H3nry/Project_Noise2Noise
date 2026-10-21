from model import *
import torch


"""
######################
Test Conv2d
######################
"""
def test_conv2d():
    in_channels = 3
    input = torch.rand(50, in_channels, 32, 32)

    out_channels = 32
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (0, 0)
    bias = True
    dilation = (1, 1)

    torch_conv = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
    self_conv = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
    self_conv.weight = torch_conv.weight
    self_conv.bias = torch_conv.bias

    result_torch = torch_conv(input)
    result_own = self_conv(input)
    torch.testing.assert_close(result_torch, result_own)

"""
######################
Test TranposeConv2d
######################
"""
def test_transposeConv2d():
    in_channels=3
    input = torch.rand(50,in_channels,32,32)

    out_channels=32
    kernel_size=(3,3)
    stride=(2,2)
    padding=(0,0)
    output_padding = (0,0)
    bias=True
    dilation=(1,1)

    torch_transconv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
    own_transconv = TransposeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)

    own_transconv.weight = torch_transconv.weight
    own_transconv.bias = torch_transconv.bias
    result_torch = torch_transconv(input)
    result_own = own_transconv(input)
    torch.testing.assert_close(result_torch, result_own)

"""
######################
Test backward Conv2d
######################
"""
@torch.enable_grad()
def test_backward_conv2d():
    torch.set_default_dtype(torch.float64)
    class torch_Net(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding,dilation):
            super(torch_Net,self).__init__()
        
            
            self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        
        def forward(self, input):
            input = self.conv1(input)
            return input

    in_channels=24
    out_channels=3
    kernel_size=(5,4)
    stride=(1,3)
    padding=(0,0)
    dilation=(1,2)
    bias=True

    input = torch.randn(15, in_channels,28,32, requires_grad=True)
    my_nn = torch_Net(in_channels, out_channels, kernel_size, stride, padding, dilation)

    out_torch = my_nn(input)

    my_nn.zero_grad()
    initial_gradient = torch.rand_like(out_torch)
    out_torch.backward(initial_gradient)
    torch_gradient = input.grad
    torch_gradient_weight = my_nn.conv1.weight.grad
    if bias:
        torch_gradient_bias = my_nn.conv1.bias.grad

    with torch.no_grad():
        own_conv2d = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        own_conv2d.weight = my_nn.conv1.weight
        own_conv2d.bias = my_nn.conv1.bias

        out_own = own_conv2d(input)
        own_gradient=own_conv2d.backward(initial_gradient)
        own_gradients = own_conv2d.param()
    
    try:
        assert(own_gradient.shape==torch_gradient.shape)
        torch.testing.assert_close(out_torch, out_own)
        torch.testing.assert_close(torch_gradient, own_gradient)
        torch.testing.assert_close(torch_gradient_weight, own_gradients[0][1])

        if bias:
            torch.testing.assert_close(torch_gradient_bias, own_gradients[1][1])
    finally:
        torch.set_default_dtype(torch.float32)

"""
######################
Test backward Transpose Conv2d
######################
"""
@torch.enable_grad()
def test_backward_transposeConv2d():
    torch.set_default_dtype(torch.float64)
    class torch_Net(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation,bias):
            super(torch_Net,self).__init__()


            self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_padding,dilation=dilation,bias=bias)

        def forward(self, input):
            input = self.conv1(input)
            return input

    in_channels=10
    out_channels=12
    kernel_size=(3,4)
    stride=(3,4)
    padding=(2,3)
    output_padding = (1,0)
    dilation=(2,1)
    bias=True

    input = torch.randn(20,in_channels,15,15, requires_grad=True)
    my_nn = torch_Net(in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation,bias)

    out_torch = my_nn(input)

    my_nn.zero_grad()
    initial_gradient = torch.rand_like(out_torch)
    out_torch.backward(initial_gradient)
    torch_gradient = input.grad
    torch_gradient_weight = my_nn.conv1.weight.grad
    if bias:
        torch_gradient_bias = my_nn.conv1.bias.grad

    with torch.no_grad():
        own_conv2d = TransposeConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_padding,dilation=dilation,bias=bias)
        own_conv2d.weight = my_nn.conv1.weight
        own_conv2d.bias = my_nn.conv1.bias

        out_own = own_conv2d(input)
        own_gradient=own_conv2d.backward(initial_gradient)
        own_gradients = own_conv2d.param()

    try:
        assert(own_gradient.shape==torch_gradient.shape)
        torch.testing.assert_close(out_torch,out_own)
        torch.testing.assert_close(torch_gradient , own_gradient)
        torch.testing.assert_close(torch_gradient_weight , own_gradients[0][1])
        if bias:
            torch.testing.assert_close(torch_gradient_bias, own_gradients[1][1])
    finally:
            torch.set_default_dtype(torch.float32)

"""
######################
Test network with own methods
######################
"""
def test_network():
    class torch_Net(torch.nn.Module):
        def __init__(self):
            super(torch_Net,self).__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(1,1,(3,3),2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(1,1,(3,3),2),
                torch.nn.ConvTranspose2d(1,1,(3,3)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(1,1,(3,3)),
                torch.nn.Sigmoid()
            )
        def forward(self, input):
            return self.seq(input)
        
    input = torch.randn((3,1,10,10))
    net = torch_Net()

    out_torch = net(input)

    class own_Net():
        def __init__(self):
            self.seq = Sequential(
                Conv2d(1,1,(3,3),2),
                ReLU(),
                Conv2d(1,1,(3,3),2),
                TransposeConv2d(1,1,(3,3)),
                ReLU(),
                TransposeConv2d(1,1,(3,3)),
                Sigmoid()
            )
        def forward(self, input):
            return self.seq(input)
    own_net = own_Net()
    out = own_net.forward(input)
    assert(out_torch.shape==out.shape)

"""
######################
Test ReLU forward and backward
######################
"""
@torch.enable_grad()
def test_relu():
    class torch_Net(torch.nn.Module):
        def __init__(self):
            super(torch_Net,self).__init__()
                
            self.relu = torch.nn.ReLU()
        
        def forward(self, input):
            output = self.relu(input)
            return output

    in_channels=15
    input = torch.randn(20,in_channels,15,15, requires_grad=True)
    my_nn = torch_Net()

    out_torch = my_nn(input)

    my_nn.zero_grad()
    initial_gradient = torch.rand_like(out_torch)
    out_torch.backward(initial_gradient)
    torch_gradient = input.grad

    with torch.no_grad():
        own_relu = ReLU()

        out_own = own_relu(input)
        own_gradient=own_relu.backward(initial_gradient)

    assert(own_gradient.shape == torch_gradient.shape)
    torch.testing.assert_close(torch_gradient, own_gradient)
    torch.testing.assert_close(out_torch, out_own)

"""
######################
Test sigmoid forward and backward
######################
"""
@torch.enable_grad()
def test_sigmoid():
    class torch_Net(torch.nn.Module):
        def __init__(self):
            super(torch_Net,self).__init__()
           
            self.sig = torch.nn.Sigmoid()
        
        def forward(self, input):
            output = self.sig(input)
            return output

    in_channels=15
    input = torch.randn(20,in_channels,15,15, requires_grad=True)
    my_nn = torch_Net()

    out_torch = my_nn(input)

    my_nn.zero_grad()
    initial_gradient = torch.rand_like(out_torch)

    out_torch.backward(initial_gradient)
    torch_gradient = input.grad

    with torch.no_grad():
        own_sig = Sigmoid()

        out_own = own_sig(input)
        own_gradient=own_sig.backward(initial_gradient)

    assert(own_gradient.shape==torch_gradient.shape)
    torch.testing.assert_close(torch_gradient , own_gradient)
    torch.testing.assert_close(out_torch,out_own)

"""
######################
Test MSE forward and backward
######################
"""
@torch.enable_grad()
def test_MSE():
    class torch_Net(torch.nn.Module):
        def __init__(self, reduction):
            super(torch_Net,self).__init__()
    
            self.mse = torch.nn.MSELoss(reduction=reduction)
        
        def forward(self, input, target):
            output = self.mse(input, target)
            return output

    in_channels=15
    input = torch.randn(20,in_channels,12,15, requires_grad=True)
    target = torch.rand_like(input)
    reduction = 'mean' # The only reduction implemented in custom MSE is "mean"
    my_nn = torch_Net(reduction)

    out_torch = my_nn(input, target)

    my_nn.zero_grad()
    out_torch.backward()

    torch_gradient = input.grad

    with torch.no_grad():
        own_mse = MSE()

        out_own = own_mse(input, target)
        own_gradient=own_mse.backward()


    assert(own_gradient.shape == torch_gradient.shape)
    torch.testing.assert_close(torch_gradient, own_gradient)
    torch.testing.assert_close(out_torch, out_own)

"""
######################
Test Sequential forward and backward
######################
"""
@torch.enable_grad()
def test_sequential():
    class torch_Net(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding,dilation,bias):
            super(torch_Net,self).__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation),
                torch.nn.Sigmoid()
            )

        def forward(self, input):
            return self.seq(input)
        
    in_channels=15
    out_channels=12
    kernel_size=(5,7)
    stride=(3,4)
    padding=(2,2)
    dilation=(1,2)
    bias=True

    input = torch.randn(15,in_channels,20,20, requires_grad=True)
    
    my_nn = torch_Net(in_channels, out_channels, kernel_size, stride, padding,dilation,bias)

    out_torch = my_nn(input)

    my_nn.zero_grad()
    initial_gradient = torch.rand_like(out_torch)
    out_torch.backward(initial_gradient)
    torch_gradient = input.grad

    with torch.no_grad():
        own_seq = Sequential(
                    Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation),
                    Sigmoid()
                )

        own_seq.args[0].weight = my_nn.seq[0].weight
        own_seq.args[0].bias = my_nn.seq[0].bias
        
        out_own = own_seq(input)
        own_gradient=own_seq.backward(initial_gradient)

    assert(own_gradient.shape==torch_gradient.shape)
    torch.testing.assert_close(out_torch,out_own)
    torch.testing.assert_close(torch_gradient , own_gradient)

"""
######################
Test SGD
######################
"""
@torch.enable_grad()
def test_sgd():
    torch.set_default_dtype(torch.float64)
    class torch_Net(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding,dilation,bias):
            super(torch_Net,self).__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias),
                torch.nn.Sigmoid()
            )
        def forward(self, input):
            return self.seq(input)
        
    in_channels=3
    out_channels=24
    kernel_size=(5,7)
    stride=(2,3)
    padding=(1,2)
    dilation=(1,2)
    bias=True
    learning_rate = 0.05
    
    # Torch net
    torch_nn = torch_Net(in_channels, out_channels, kernel_size, stride, padding,dilation,bias)
    torch_loss_fn = torch.nn.MSELoss()
    
    # own net
    own_nn = Sequential(
                Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias),
                Sigmoid()
            )
    own_nn.args[0].weight = torch_nn.seq[0].weight.detach().clone()
    if bias:
        own_nn.args[0].bias = torch_nn.seq[0].bias.detach().clone()
    own_loss_fn = MSE()
    
    # optimizer
    torch_optimizer = torch.optim.SGD(torch_nn.parameters(),lr=learning_rate)
    
    epoch = 50
    for _ in range(epoch):
        input = torch.randn(15,in_channels,20,20, requires_grad=True)
        
        # torch model
        torch_optimizer.zero_grad()
        torch_prediction = torch_nn(input)
        target = torch.randn_like(torch_prediction)
        torch_loss = torch_loss_fn(torch_prediction,target)
        torch_loss.backward()

        torch_optimizer.step()

        
        with torch.no_grad(): # own model
            own_prediction = own_nn(input)
            _ = own_loss_fn(own_prediction,target)
            gradwrtoutput = own_loss_fn.backward()
            own_nn.backward(gradwrtoutput)
            own_optimizer = SGD(own_nn.param(),lr=learning_rate)

            own_optimizer.step()
    try:       
        torch.testing.assert_close(own_nn.args[0].weight,torch_nn.seq[0].weight)
        if bias:
            torch.testing.assert_close(own_nn.args[0].bias,torch_nn.seq[0].bias)
    finally:
        torch.set_default_dtype(torch.float32)
"""
######################
Call tests
######################
"""
test_conv2d()
test_transposeConv2d()
test_backward_conv2d()
test_backward_transposeConv2d()
test_network()
test_relu()
test_sigmoid()
test_MSE()
test_sequential()
test_sgd()

print("All tests passed successfully!")