import numpy as np
import torch
from sklearn.linear_model import LinearRegression

def fit_params(
    x,
    y,
    fun,
    a_range=(-10, 10),
    b_range=(-10, 10),
    grid_number=101,
    iteration=3,
    verbose=True,
):
    """
    fit a, b, c, d such that

    .. math::
        |y-(cf(ax+b)+d)|^2

    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.

    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True

    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)

    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    """
    # ----
    
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing="ij")
        post_fun = fun(
            a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :]
        )
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = (
            torch.sum((post_fun - x_mean) * (y - y_mean)[:, None, None], dim=0)
            ** 2
        )
        denominator = torch.sum((post_fun - x_mean) ** 2, dim=0) * torch.sum(
            (y - y_mean)[:, None, None] ** 2, dim=0
        )
        r2 = numerator / (denominator + 1e-4)
        r2 = torch.nan_to_num(r2)

        best_id = torch.argmax(r2)
        a_id, b_id = (
            torch.div(best_id, grid_number, rounding_mode="floor"),
            best_id % grid_number,
        )

        if (
            a_id == 0
            or a_id == grid_number - 1
            or b_id == 0
            or b_id == grid_number - 1
        ):
            if _ == 0 and verbose:
                print("Best value at boundary.")
            if a_id == 0:
                a_arange = [a_[0], a_[1]]  # noqa
            if a_id == grid_number - 1:
                a_arange = [a_[-2], a_[-1]]  # noqa
            if b_id == 0:
                b_arange = [b_[0], b_[1]]  # noqa
            if b_id == grid_number - 1:
                b_arange = [b_[-2], b_[-1]]  # noqa

        else:
            a_range = [a_[a_id - 1], a_[a_id + 1]]
            b_range = [b_[b_id - 1], b_[b_id + 1]]

    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]


    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(
        post_fun[:, None].detach().numpy(), y.detach().numpy()
    )
    c_best = torch.from_numpy(reg.coef_)[0]
    d_best = torch.from_numpy(np.array(reg.intercept_))
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best




# unit test cases
x = torch.linspace(-1, 1, steps=100)
y = 5.0 * torch.sin(3.0 * x + 2.0) + 0.7 + torch.normal(0, 1, (100,)) * 0.02
fit_params(x, y, torch.sin)
print(fit_params(x, y, torch.sin))


x = torch.linspace(-2, 2, steps=50)
y = 4.0 * torch.cos(2.0 * x + 1.0) + 1.0 + torch.normal(0, 1, (50,)) * 0.05
fit_params(
            x, y, torch.cos, a_range=(-2, 2), b_range=(-2, 2), grid_number=5
        )
print(fit_params(x, y, torch.sin))

x = torch.linspace(-3, 3, steps=150)
y = 2.0 * torch.tanh(0.5 * x + 3.0) + 0.5 + torch.normal(0, 1, (150,)) * 0.1
fit_params(
            x, y, torch.tanh, iteration=5
        )
print(fit_params(x, y, torch.sin))