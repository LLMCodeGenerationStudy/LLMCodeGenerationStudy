import torch
import torch.nn.functional as F


def layer_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    """
    Applies Layer Normalization to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    weight : torch.Tensor
        The weight tensor for scaling the normalized input.
    bias : torch.Tensor or None
        The bias tensor for shifting the normalized input. If None, no bias is added.
    residual : torch.Tensor or None
        The residual tensor to be added to the normalized input. If None, no residual is added.
    eps : float, optional
        A small constant added for numerical stability.
    prenorm : bool, optional
        If True, the function returns both the normalized output and the original input.
    upcast : bool, optional
        If True, upcasts all tensors to float type for computation.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The layer-normalized output tensor. If `prenorm` is True, returns a tuple
        containing the output tensor and the original input tensor.

    Notes
    -----
    Layer Normalization normalizes the input tensor by subtracting the mean and dividing by
    the standard deviation of each individual layer. This function performs the normalization,
    applies the weight and bias, and optionally adds a residual connection. If `upcast` is True,
    all tensors are upcast to float type before the operations for precision.
    """
    # ----

    # Preserve the original dtype
    dtype = x.dtype

    # Upcast weight, bias, and x to float if necessary
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        x = x.float()
        residual = residual.float() if residual is not None else residual

    # Add residual to input if provided
    if residual is not None:
        x = (x + residual).to(dtype)

    # Apply layer normalization
    # Note: It was x.to(weight.dtype) but it should be x because F.layer_norm expects the input tensor x.
    out = F.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps).to(dtype)

    # Return the output only or the output with the original input if prenorm is True
    return out if not prenorm else (out, x)


# unit test cases
# print(layer_norm_ref(torch.rand(10, 5)))


# unit test cases
print(layer_norm_ref(torch.rand(10, 5), torch.rand(5,), torch.rand(5,)))
print(layer_norm_ref(torch.rand(10, 5), torch.rand(5,), torch.rand(5,), torch.rand(10, 5), 1e-5, False, True))
print(layer_norm_ref(torch.rand(10, 5), torch.rand(5,), None, None, 1e-6, True, False))
print(layer_norm_ref(torch.rand(10, 5), torch.rand(5,), None, torch.rand(5,), 1e-6, True, False))
print(layer_norm_ref(torch.rand(10, 5), torch.rand(5,), None))