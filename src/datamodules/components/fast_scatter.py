from functools import partial

from src.models.scatter import FastScatterW1, Scatter, Scatter_Diffuse_Second, ScatterW1


class FastScatterTransform:
    def __init__(self, device="cpu", agg="moment", alpha=0, power=2, cheb_order=None):
        self.device = device
        self.agg = agg
        self.alpha = alpha
        self.power = power
        self.cheb_order = cheb_order
        assert power == 1 or power == 2
        if power == 2:
            self.scatter_fn = Scatter
        elif cheb_order is not None and cheb_order != "None":
            print("Fast scatterw1")
            self.scatter_fn = partial(FastScatterW1, cheb_order=cheb_order)
        else:
            self.scatter_fn = ScatterW1

    def __call__(self, data):
        data.x = self.scatter_fn(in_channels=data.x.shape[1], agg=self.agg, alpha=self.alpha)(data)
        return data


class FastScatterTransformSort:
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, data):
        data.x = Scatter_Diffuse_Second(data.x.shape[1])(data)
        return data
