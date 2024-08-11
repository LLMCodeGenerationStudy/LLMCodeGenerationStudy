import torch
import torch.nn as nn


def fuse_deconv_and_bn(deconv, bn):
    """
    Fuses a ConvTranspose2d layer with a BatchNorm2d layer into a single ConvTranspose2d layer.

    Arguments:
    deconv : torch.nn.ConvTranspose2d
        The convolutional transpose layer to be fused.
    bn : torch.nn.BatchNorm2d
        The batch normalization layer to be fused.

    Returns:
    torch.nn.ConvTranspose2d
        The resulting fused ConvTranspose2d layer.
    """
    # ----
    
    # Create a new ConvTranspose2d layer with the same parameters as the original but with bias enabled
    fuseddconv = nn.ConvTranspose2d(
        in_channels=deconv.in_channels,
        out_channels=deconv.out_channels,
        kernel_size=deconv.kernel_size,
        stride=deconv.stride,
        padding=deconv.padding,
        output_padding=deconv.output_padding,
        dilation=deconv.dilation,
        groups=deconv.groups,
        bias=True  # Always true to accommodate the fused bias
    ).requires_grad_(False).to(deconv.weight.device)  # Disable grad and move to the same device

    # Prepare the weights for fusion
    w_deconv = deconv.weight.clone().view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(
        torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))  # Apply the BN transformation to weights

    # Prepare and fuse the biases
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)  # Combine biases from BN and deconv layers

    return fuseddconv


# unit test cases
deconv = nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1, dilation=1, groups=1, bias=None)
bn = nn.BatchNorm2d(3)
fused_layer = fuse_deconv_and_bn(deconv, bn)
print(fused_layer)

initial_bias = torch.randn(3)
deconv = nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1, dilation=1, groups=1, bias=True)
deconv.bias.data = initial_bias.clone()
bn = nn.BatchNorm2d(3)
fused_layer = fuse_deconv_and_bn(deconv, bn)
print(fused_layer)

deconv = nn.ConvTranspose2d(3, 4, (5, 5), (2, 2), (1, 1), output_padding=(1, 1), dilation=(2, 2))
bn = nn.BatchNorm2d(4)
fused_layer = fuse_deconv_and_bn(deconv, bn)
print(fused_layer)

