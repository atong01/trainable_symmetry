from src.models.scatter2 import Scatter, Scatter_Diffuse_Second, Scatter_W1


class FastScatterTransform:
    def __init__(self, device="cpu", agg="moment", alpha=0, power=2):
        self.device = device
        self.agg = agg
        self.alpha = alpha
        self.power = power
        assert power == 1 or power == 2
        self.scatter_fn = Scatter if power == 2 else Scatter_W1

    def __call__(self, data):
        data.x = self.scatter_fn(data.x.shape[1], agg=self.agg, alpha=self.alpha)(data)
        return data


class FastScatterTransformSort:
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, data):
        data.x = Scatter_Diffuse_Second(data.x.shape[1])(data)
        return data
