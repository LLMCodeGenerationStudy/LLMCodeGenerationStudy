import torch


def rms_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    """
    Applies Residual Multi-scale Normalization (RMSNorm) to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    weight : torch.Tensor
        The weight tensor for scaling the normalized input.
    bias : torch.Tensor or None
        The bias tensor for shifting the normalized input. If None, no bias is added.
    residual : torch.Tensor or None
        The residual tensor to be added to the input. If None, no residual is added.
    eps : float, optional
        A small constant added for numerical stability.
    prenorm : bool, optional
        If True, the function returns both the normalized output and the original input.
    upcast : bool, optional
        If True, upcasts all tensors to float type for computation.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The RMS normalized output tensor. If `prenorm` is True, returns a tuple
        containing the output tensor and the original input tensor.

    Notes
    -----
    RMSNorm is a normalization technique that scales the input by the inverse square root
    of the mean of squares. This function performs the normalization, applies the weight
    and bias, and optionally adds a residual connection. If `upcast` is True, the
    function upcasts all tensors to float type before performing the operations to
    maintain precision.
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
        x = (x + residual).to(x.dtype)

    # Compute the reciprocal of the standard deviation of the input
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)

    # Apply normalization, weight, and bias
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)

    # Cast the output back to the original dtype
    out = out.to(dtype)

    # Return the output only or the output with the original input if prenorm is True
    return out if not prenorm else (out, x)


# unit test cases
print(rms_norm_ref(torch.rand(10, 5), torch.rand(5,), torch.rand(5,), None))
print(rms_norm_ref(torch.rand(10, 5), torch.rand(5,), torch.rand(5,), None, 1e-5, False, True))
print(rms_norm_ref(torch.rand(10, 5), torch.rand(5,), torch.rand(5,), torch.rand(10, 5)))
print(rms_norm_ref(torch.rand(10, 5), torch.rand(5,), None, torch.rand(5,), 1e-6, True, False))
print(rms_norm_ref(torch.rand(10, 5), torch.rand(5,), None))