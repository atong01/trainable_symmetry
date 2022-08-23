from src.models.scatter import Scatter, Scatter_Diffuse_Second


class FastScatterTransform:
    def __init__(self, device="cpu", agg="moment", alpha=0):
        self.device = device
        self.agg = agg
        self.alpha = alpha

    def __call__(self, data):
        data.x = Scatter(data.x.shape[1], agg=self.agg, alpha=self.alpha)(data)
        return data


class FastScatterTransformSort:
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, data):
        data.x = Scatter_Diffuse_Second(data.x.shape[1])(data)
        return data
