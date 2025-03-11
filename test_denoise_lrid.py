import os

os.environ["OPENMP_NUM_THREADS"] = "16"

import argparse
import random
import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import *
from utils.imgproc import *

from datasets.real_dataset import LRIDEvalDataset
from models.ELD_models import UNetSeeInDark


def build_model(args):
    model = UNetSeeInDark().to(args.device)
    model.load_state_dict(torch.load(args.cp_dir, map_location="cpu"), strict=True)
    model.eval()
    return model


def build_dataloader(args):
    valid_set = LRIDEvalDataset(
        clip_low=False, clip_high=True, ratio_list=args.eval_ratio, dataset_name=args.evalset_type
    )

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2)
    return valid_loader


def valid_one_ep(model, valid_loader, args, plot_res=False):
    psnr_am = AverageMeter("valid_psnr", ":.2f")
    ssim_am = AverageMeter("valid_ssim", ":.4f")

    for data_id, data in enumerate(tqdm(valid_loader)):
        imgs_hr, imgs_lr = tensor_dim5to4(data["hr"]).to(args.device), tensor_dim5to4(data["lr"]).to(args.device)

        imgs_dn = model(imgs_lr)

        imgs_dn, imgs_hr = torch.clamp(imgs_dn, 0, 1), torch.clamp(imgs_hr, 0, 1)
        pmn_metric_dict = PMN_metric(imgs_dn, imgs_hr)
        psnr_am.update(pmn_metric_dict["psnr"].item())
        ssim_am.update(pmn_metric_dict["ssim"].item())

        ## convert raw to rgb for plotting
        if plot_res:
            wb, ccm, iso = data["wb"][0].numpy(), data["ccm"][0].numpy(), int(data["iso"].squeeze().item())
            for x, y in zip([imgs_lr[0], imgs_dn[0], imgs_hr[0]], ["inputs", "pred", "gt"]):
                x = x.clamp_(0, 1).detach().cpu().permute(1, 2, 0).numpy()
                x = rggb_to_srgb(x, wb=wb, ccm=ccm, gamma=3, format="rggb", uint8=True)
                imageio.imwrite(f"./test_res/{args.task}/{data_id}_{y}_iso{iso}.png", x)
    return psnr_am.avg, ssim_am.avg


def main(args):
    model = build_model(args)
    valid_loader = build_dataloader(args)

    delete_and_remake_dir(f"./test_res/{args.task}")

    valid_psnr, valid_ssim = valid_one_ep(model, valid_loader, args, plot_res=args.plot_res)
    print(f"valid_psnr: {valid_psnr}, valid_ssim: {valid_ssim}")


##--------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## common
    parser.add_argument("--task", type=str, default="lrid")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--cp_dir", type=str, default="./checkpoints/imx686.pth")
    ## change below for different setups
    parser.add_argument("--plot_res", type=bool, default=False)
    parser.add_argument("--evalset_type", type=str, default="indoor_x5", choices=["indoor_x5", "outdoor_x3"])
    parser.add_argument(
        "--eval_ratio",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="[1, 2, 4, 8, 16] for indoor_x5, [1, 2, 4] for outdoor_x3",
    )
    _args = parser.parse_args(args=[])

    # fix seed
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    random.seed(_args.seed)
    torch.backends.cudnn.benchmark = True

    main(_args)
