import torch
import sys
sys.path.append("../Pretrained_IPT")
# from Pretrained_IPT import Pretrained_IPT.model as IPTmodel
from Pretrained_IPT.option import args
import Pretrained_IPT.model as IPTmodel
import Pretrained_IPT.utility as utility
import DeRaindrop.models
from  DeRaindrop.predict import align_to_four, predict
from MPRNet.Deraining.MPRNet import MPRNet
import torch.nn.functional as F
# from .. import Pretrained_IPT.model as IPTmodel


def get_model(model_name):
    if model_name.lower() == "ipt":
        args.cpu = True
        args.derain = True
        args.test_only = True
        args.pretrain = "../IPT_derain.pt"
        args.save = "./"
        args.scale = [1]
        print(args)
        checkpoint = utility.checkpoint(args)
        model = IPTmodel.Model(args, checkpoint)
        state_dict = torch.load(args.pretrain, map_location=torch.device('cpu'))
        model.model.load_state_dict(state_dict,)
        return model
    elif model_name.lower() == "attentive_gan":
        model = DeRaindrop.models.generator.Generator()
        model.load_state_dict(torch.load('../DeRaindrop/weights/gen.pkl', map_location=torch.device('cpu')))
        return model
    elif model_name.lower() == "mpr_net":
        model = MPRNet()

        ckpt = torch.load("../model_deraining.pth", map_location=torch.device('cpu'))
        try:
            model.load_state_dict(ckpt["state_dict"])
        except:
            state_dict = ckpt["state_dict"]
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        return model
    elif model_name.lower() == "derain_net":
        pass


def predict_img(model, img, model_name):

    if model_name.lower() == "ipt":
        input_ = img
        input_ = input_.reshape(1,input_.shape[0],input_.shape[1],input_.shape[2])
        return model(input_, 0)
    elif model_name.lower() == "attentive_gan":
        img = align_to_four(img)

        c,h,w = img.shape
        img = img.reshape(1, c, h, w)
        img = img.to("cpu")
        model = model.to('cpu')
        result = model(img)
        print(result[-1])
        return result[-1]
    elif model_name.lower() == "mpr_net":
        input_ = img
        img_multiple_of = 8
        h, w = input_.shape[1], input_.shape[2]
        H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw,0, padh), 'reflect')

        with torch.no_grad():
            restored = model(input_.reshape(1,input_.shape[0],input_.shape[1],input_.shape[2]))
        restored = restored[0]

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :h, :w]
        return restored
    elif model_name.lower() == "derain_net":
        pass









