import torch
from backend import model_utils
import os
import numpy as np
from PIL import Image
import torchvision
from skimage.metrics import structural_similarity

def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

models = ["attentive_gan", "mpr_net"]
files = os.listdir("../Rain100L/rainy")
print(files)
selected_files = np.random.choice(files, size=10, replace=False)

selected_gt = ["../Rain100L/no"+ file for file in selected_files]
selected_files = ["../Rain100L/rainy/"+ file for file in selected_files]
for i in range(2):
    model = model_utils.get_model(models[i])
    psnr = 0
    ssim = 0
    for j in range(10):
        gt = Image.open(selected_gt[j])
        gt = gt.convert('RGB')
        gt = torchvision.transforms.functional.pil_to_tensor(gt) / 255.

        input = Image.open(selected_files[j])
        input = input.convert('RGB')
        input = torchvision.transforms.functional.pil_to_tensor(input) / 255.

        output = model_utils.predict_img(model, input, model_name=models[i])
        output = output.clamp(0,1)
        c,h,w = gt.shape
        print(output.shape)
        print(gt.shape)
        output = torchvision.transforms.functional.resize(output, (h,w))
        output = output.squeeze()
        # output =
        # psnr += PSNR(gt*255, output*255)
        ssim += structural_similarity(gt.detach().numpy(), output.detach().numpy(),channel_axis=0,data_range=1.0)
    print(ssim, models[i])




