import torch
from scipy import special, pi

def logbessel_I_scipy(nu, z, check = True):
	'''
	Pytorch version of scipy computation of modified Bessel functions 
	of the 1st kind I(nu,z).
	Parameters
	----------
	nu: positive int, float
		Order of modified Bessel function of 1st kind.
	z: int/float or tensor, shape (N,) 
		Argument of Bessel function.
	check: bool
		If True, check if argument of log is non zero.
	
	Return
	------
	result: tensor, shape (N,)
	'''
    # ----

	if not isinstance(z, torch.Tensor):
		z = torch.tensor(z)
	z = z.reshape(-1)

	result = special.ive(nu, z)
	if check:
		assert len(result[ result == 0]) == 0
	result = torch.log(result) + z
	return result

# unit test cases
assert(logbessel_I_scipy(nu=1, z=5.0) == logbessel_I_scipy(nu=1, z=5.0))
assert(logbessel_I_scipy(nu=0.5, z=torch.tensor([0.1, 1.0, 10.0])) == logbessel_I_scipy(nu=0.5, z=torch.tensor([0.1, 1.0, 10.0])))
assert(logbessel_I_scipy(nu=-0.0001, z=0.0001) == logbessel_I_scipy(nu=-0.0001, z=0.0001))