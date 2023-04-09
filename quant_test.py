import torch
from torch import optim
import torchvision
import utils
import model_io
from models.unet_adaptive_bins import UnetAdaptiveBins

if __name__ == '__main__':
    model = UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=10)
    pretrained_path = "./checkpoints/UnetAdaptiveBins_02-Apr_22-04-nodebs2-tep4-lr0.000357-wd0.1-584ce456-69f8-436a-a71b-4abd296241f9_latest.pt"
    model, opt, ep = model_io.load_checkpoint(pretrained_path, model)
    backend = "fbgemm"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    params = [{"params": model.get_1x_lr_params(), "lr": 1e-3 / 10},
            {"params": model.get_10x_lr_params(), "lr": 1e-3}]
    optimizer = optim.AdamW(params)
    optimizer.load_state_dict(opt)
    model_io.save_checkpoint(model_static_quantized, optimizer, ep, 'quant.pt', '.')
