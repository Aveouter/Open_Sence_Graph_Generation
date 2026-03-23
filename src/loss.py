import torch.nn as nn


LOSS_FACTORY = {
    "ce": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "bce": nn.BCEWithLogitsLoss,
    "hstrnet_loss": lambda: HSTRNet_Loss(),  # 这里假设你有一个自定义的 HSTRNet_Loss 类

}

def loss_construction(loss_name="ce"):
    try:
        return LOSS_FACTORY[loss_name]()  # 调用构造函数
    except KeyError:
        raise ValueError(f"Unknown loss type: {loss_name}")
    

def HSTRNet_Loss():
    # 这里是你自定义的 HSTRNet_Loss 的实现
    # 你可以根据需要添加参数和逻辑
    class _HSTRNet_Loss(nn.Module):
        def __init__(self):
            super().__init__()
            # 初始化你的损失函数组件

        def forward(self, outputs, targets):
            # 实现你的损失计算逻辑
            loss = 1  # 计算损失
            return loss

    return _HSTRNet_Loss()