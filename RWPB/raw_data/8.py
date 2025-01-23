import numpy as np
import torch

def cvimg2torch(img):
    '''Convert a img to tensor
    input:
        im -> ndarray uint8 HxWxC 
    return
        tensor -> torch.tensor BxCxHxW 
    '''
    # ----

    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img



# unit test cases
print(cvimg2torch((np.random.rand(256, 256, 3) * 255).astype(np.uint8)))
print(cvimg2torch(np.repeat((np.random.rand(10, 10) * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)))
print(cvimg2torch(np.array([[[300, -10, 500], [256, 255, 0]]], dtype=np.int32)))
