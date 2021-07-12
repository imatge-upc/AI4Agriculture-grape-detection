import torch

class UnNormalize(object):
    def __init__(self, image_mean, image_std):
        self.mean = torch.tensor(image_mean)
        self.std = torch.tensor(image_std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.std.device != tensor.device:
            self.std=self.std.to(tensor.device)
            self.mean=self.mean.to(tensor.device)
        return (tensor * self.std) + self.mean
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
