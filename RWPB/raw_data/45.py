import torch

def invalid_to_zeros(arr, valid_mask, ndim=999):
    """
    Replaces invalid entries in an array with zeros, based on a validity mask.

    Parameters
    ----------
    arr : torch.Tensor
        The input tensor with potentially invalid entries.
    valid_mask : torch.Tensor (boolean)
        A boolean mask with the same number of elements as 'arr', where True indicates
        a valid entry and False indicates an invalid entry that should be replaced with zero.
    ndim : int, optional
        The desired number of dimensions for the output tensor. If 'arr' has more than
        'ndim' dimensions, it will be flattened accordingly. Default is 999, meaning no flattening.

    Returns
    -------
    torch.Tensor
        The input tensor 'arr' with invalid entries replaced by zeros.
    int or torch.Tensor
        The number of non-zero entries in the validity mask (nnz), which corresponds to the
        number of valid points per image.

    Notes
    -----
    This function is particularly useful in the context of point cloud processing or similar
    applications where some data points may be invalid or missing. By setting these invalid
    points to zero, the resulting tensor can be used for further processing, such as neural
    network input.

    If 'valid_mask' is provided, the function updates 'arr' in-place to zero out the invalid
    entries. It also calculates the number of non-zero entries in the mask (nnz), which is
    returned separately. If 'valid_mask' is None, it assumes all entries are valid and calculates
    nnz based on the total number of elements in 'arr' divided by the first dimension size.

    If 'arr' has more dimensions than specified by 'ndim', the function flattens it to the
    desired number of dimensions.
    """
    # ----

    if valid_mask is not None:
        arr = arr.clone()  # Create a copy of arr to avoid modifying the original tensor
        arr[~valid_mask] = 0  # Set invalid entries (where mask is False) to zero
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)  # Calculate the number of valid points per image
    else:
        # If no valid_mask is provided, calculate nnz based on total elements
        nnz = arr.numel() // len(arr) if len(arr) else 0

    # If the number of dimensions in arr is greater than ndim, flatten it
    if arr.ndim > ndim:
        if ndim > 0:
            arr = arr.reshape([arr.shape[0], -1])[:ndim]
        else:
            arr = arr.flatten()

    return arr, nnz


# unit test cases
print(invalid_to_zeros(torch.tensor([1, 2, 3, 4]), valid_mask = torch.tensor([True, False, True, True])))
print(invalid_to_zeros(torch.tensor([1, 2, 3, 4]), torch.tensor([False, False, False, False])))
print(invalid_to_zeros(torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[True, True], [False, True]], [[True, False], [True, True]]]), 1))