import torch

def reshape_tensor(x, heads):
    """
    Reshape a tensor to fit a multi-head attention mechanism structure.
    If the shape of x is invalid, please return ValueError.

    Arguments:
    x : torch.Tensor
        The input tensor with shape (batch_size, length, width), where 'width'
        is typically the embedding dimension or a multiple of 'heads'.
    heads : int
        The number of heads in the multi-head attention mechanism.

    Returns:
    torch.Tensor
        A reshaped tensor with dimensions corresponding to multi-head structures,
        specifically (batch_size * n_heads, length, dim_per_head).
    """
    # ----

    bs, length, width = x.shape
    # Reshape to introduce a new dimension for heads, assuming that width can be evenly divided by heads.
    # width is divided into 'heads' and the dimension per head is calculated as width // heads
    try:
        x = x.view(bs, length, heads, -1)
    except:
        return ValueError
    # (bs, length, heads, dim_per_head) --> (bs, heads, length, dim_per_head)
    # Transpose to swap the 'length' and 'heads' dimensions to align heads for parallel processing
    x = x.transpose(1, 2)
    # (bs, heads, length, dim_per_head) --> (bs * heads, length, dim_per_head)
    # Reshape to merge batch size and heads for processing each head independently in subsequent operations
    x = x.reshape(bs * heads, length, -1)
    return x


# unit test cases
print(reshape_tensor(torch.rand(2, 5, 10), heads=2))
print(reshape_tensor(torch.rand(3, 8, 15), heads=4))
print(reshape_tensor(torch.rand(1, 1, 4), heads=2))