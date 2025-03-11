import pickle as pkl
import torch
import numpy as np
import rawpy
from tqdm import tqdm

from torch.utils.data import Dataset


def random_crop(img, mode="non_overlap", psize=512, n_crop=8):
    res = []
    if mode == "non_overlap":
        nh, nw = img.shape[0] // psize, img.shape[1] // psize
        hs = np.random.randint(0, img.shape[0] - nh * psize + 1)
        ws = np.random.randint(0, img.shape[1] - nw * psize + 1)
        for i in range(nh):
            for j in range(nw):
                res.append(img[hs + i * psize : hs + (i + 1) * psize, ws + j * psize : ws + (j + 1) * psize, :])
    elif mode == "random":
        for i in range(n_crop):
            hs = np.random.randint(0, img.shape[0] - psize + 1)
            ws = np.random.randint(0, img.shape[1] - psize + 1)
            res.append(img[hs : hs + psize, ws : ws + psize, :])
    else:
        raise NotImplementedError
    return np.stack(res, axis=0)  # [n_crop, psize, psize, c]


def raw2bayer(raw, wl=16383, bl=512, norm=True, clip=False, format="rggb"):
    raw = raw.astype(np.float32)
    H, W = raw.shape
    r = raw[0:H:2, 0:W:2]
    gr = raw[0:H:2, 1:W:2]
    gb = raw[1:H:2, 0:W:2]
    b = raw[1:H:2, 1:W:2]
    if format == "rggb":
        out = np.stack([r, gr, gb, b], axis=-1)
    elif format == "rgbg":
        out = np.stack([r, gr, b, gb], axis=-1)
    out = (out - bl) / (wl - bl) if norm else out
    out = np.clip(out, 0, 1) if clip else out
    return out.astype(np.float32)


class SIDEvalDataset(Dataset):
    def __init__(self, wl=16383, bl=512, clip_low=False, clip_high=True, eval_ratio=250):
        super().__init__()
        self.wl, self.bl = wl, bl
        self.clip_low = 0 if clip_low else float("-inf")
        self.clip_high = 1 if clip_high else float("inf")

        ## load pmn's darkshading
        with open(f"./resources/SonyA7S2/darkshading_BLE.pkl", "rb") as f:
            self.pmn_ble = pkl.load(f)
        self.pmn_dsk_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_k.npy")
        self.pmn_dsk_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_k.npy")
        self.pmn_dsb_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_b.npy")
        self.pmn_dsb_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_b.npy")

        ## format data pairs
        with open(f"./infos/SID_evaltest.info", "rb") as info_file:
            self.data_info = pkl.load(info_file)
        self.evaltest_remap(ratio=eval_ratio)

        self.cache = {}
        for idx in tqdm(range(len(self.data_info))):
            hr_raw = np.array(rawpy.imread(self.data_info[idx]["long"]).raw_image_visible).astype(np.float32)
            lr_id = 0
            lr_raw = np.array(rawpy.imread(self.data_info[idx]["short"][lr_id]).raw_image_visible).astype(np.float32)
            self.cache[idx] = (hr_raw, lr_raw, lr_id)

    def __len__(self):
        return len(self.data_info)

    def get_darkshading(self, iso):
        if iso <= 1600:
            return self.pmn_dsk_low * iso + self.pmn_dsb_low + self.pmn_ble[iso]
        else:
            return self.pmn_dsk_high * iso + self.pmn_dsb_high + self.pmn_ble[iso]

    def lr_idremap_table_init(self):
        self.lr_idremap_table = [None] * len(self.data_info)
        for idx in range(len(self.data_info)):
            self.get_lr_id(idx)

    def get_lr_id(self, idx):
        if self.lr_idremap_table[idx] is None:
            ratio_dict = {}
            for i, ratio in enumerate(self.data_info[idx]["ratio"]):
                if ratio not in ratio_dict:
                    ratio_dict[ratio] = [i]
                else:
                    ratio_dict[ratio].append(i)
            self.lr_idremap_table[idx] = []
            for ratio in ratio_dict:
                self.lr_idremap_table[idx].append(ratio_dict[ratio])

        # 选择100, 250, 300
        ratio_id = np.random.randint(len(self.lr_idremap_table[idx]))
        id = np.random.randint(len(self.lr_idremap_table[idx][ratio_id]))
        lr_id = self.lr_idremap_table[idx][ratio_id][id]
        return lr_id

    def evaltest_remap(self, ratio=250):
        self.data_info_all = [self.data_info[:40], self.data_info[40:80], self.data_info[80:]]
        # wrapping to avoid change getitem
        for rid in range(3):
            for i in range(len(self.data_info_all[rid])):
                self.data_info_all[rid][i]["short"] = [self.data_info_all[rid][i]["short"]]
                self.data_info_all[rid][i]["ratio"] = [self.data_info_all[rid][i]["ratio"]]
        ## change eval ratio
        assert int(ratio) in [100, 250, 300], "ratio must in [100, 250, 300]"
        self.data_info = self.data_info_all[int(ratio) // 100 - 1]

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def __getitem__(self, idx):
        ## load data
        hr_raw, lr_raw, lr_id = self.cache[idx]
        dgain = self.data_info[idx]["ratio"][lr_id]

        ## subtract dark shading
        lr_raw = lr_raw - self.get_darkshading(self.data_info[idx]["ISO"])

        ## pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [h, w, c]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [h, w, c]
        lr_crops = torch.FloatTensor(lr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        hr_crops = torch.FloatTensor(hr_imgs).unsqueeze(0).permute(0, 3, 1, 2)

        lr_crops *= dgain

        data = {
            "name": f"{self.data_info[idx]['name'][:5]}_{self.data_info[idx]['ratio']}",
            "wb": torch.FloatTensor(self.data_info[idx]["wb"]),
            "ccm": torch.FloatTensor(self.data_info[idx]["ccm"]),
            "iso": self.data_info[idx]["ISO"],
            "rgb_gain": torch.ones(hr_crops.shape[0]),
            "ratio": torch.ones(hr_crops.shape[0]) * dgain,
            "lr": torch.clamp(lr_crops, self.clip_low, self.clip_high),
            "hr": torch.clamp(hr_crops, 0, 1),
        }

        return data


class ELDPairEvalDataset(Dataset):
    def __init__(self, wl=16383, bl=512, clip_low=False, clip_high=True, eval_ratio=100):
        super().__init__()
        self.wl, self.bl = wl, bl
        self.clip_low = 0 if clip_low else float("-inf")
        self.clip_high = 1 if clip_high else float("inf")
        self.eval_ratio = eval_ratio
        self.iso_list = [800, 1600, 3200]

        ## load pmn's darkshading
        with open(f"./resources/SonyA7S2/darkshading_BLE.pkl", "rb") as f:
            self.pmn_ble = pkl.load(f)
        self.pmn_dsk_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_k.npy")
        self.pmn_dsk_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_k.npy")
        self.pmn_dsb_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_b.npy")
        self.pmn_dsb_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_b.npy")

        ## data
        with open("infos/ELD_SonyA7S2.info", "rb") as info_file:
            self.data_info = pkl.load(info_file)

    def __len__(self):
        return len(self.data_info) * len(self.iso_list)

    def get_darkshading(self, iso):
        if iso <= 1600:
            return self.pmn_dsk_low * iso + self.pmn_dsb_low + self.pmn_ble[iso]
        else:
            return self.pmn_dsk_high * iso + self.pmn_dsb_high + self.pmn_ble[iso]

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def get_raw_id(self, scene_id, iso):
        for i in range(len(self.data_info[scene_id])):
            raw_iso = self.data_info[scene_id][i]["ISO"]
            if raw_iso == iso and self.eval_ratio == self.data_info[scene_id][i]["ratio"]:
                img_id = i + 1
                break
        gt_ids = np.array([1, 6, 11, 16])
        gt_id = gt_ids[np.argmin(np.abs(img_id - gt_ids))]
        return img_id - 1, gt_id - 1

    def __getitem__(self, idx):
        scene_idx, iso_idx = idx // len(self.iso_list), idx % len(self.iso_list)
        lr_id, hr_id = self.get_raw_id(scene_idx, self.iso_list[iso_idx])
        hr_raw = np.array(rawpy.imread(self.data_info[scene_idx][hr_id]["data"]).raw_image_visible).astype(np.float32)
        lr_raw = np.array(rawpy.imread(self.data_info[scene_idx][lr_id]["data"]).raw_image_visible).astype(np.float32)

        ## subtract dark shading
        lr_raw = lr_raw - self.get_darkshading(self.iso_list[iso_idx])

        ## pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [h, w, c]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [h, w, c]
        lr_crops = torch.FloatTensor(lr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        hr_crops = torch.FloatTensor(hr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        lr_crops *= self.eval_ratio

        data = {
            "name": f"scene-{idx+1:02d}_{self.data_info[scene_idx][lr_id]['name']}",
            "wb": torch.FloatTensor(self.data_info[scene_idx][hr_id]["wb"]),
            "ccm": torch.FloatTensor(self.data_info[scene_idx][hr_id]["ccm"]),
            "iso": torch.ones(hr_crops.shape[0]) * self.iso_list[iso_idx],
            "rgb_gain": torch.ones(hr_crops.shape[0]),
            "ratio": torch.ones(hr_crops.shape[0]) * self.eval_ratio,
            "lr": torch.clamp(lr_crops, self.clip_low, self.clip_high),
            "hr": torch.clamp(hr_crops, 0, 1),
        }

        return data


class LRIDEvalDataset(Dataset):
    def __init__(
        self, wl=1023, bl=64, clip_low=False, clip_high=True, ratio_list=[1, 2, 4, 8, 16], dataset_name="indoor_x3"
    ):
        super().__init__()
        self.wl, self.bl = wl, bl
        self.clip_low = 0 if clip_low else float("-inf")
        self.clip_high = 1 if clip_high else float("inf")
        self.dataset_name = dataset_name
        self.ratio_list = ratio_list
        self.iso = 6400
        self.darkshading, self.darkshading_hot = {}, {}
     
        ## format data pairs
        with open(f"infos/{dataset_name}_GT_align_ours.info", "rb") as info_file:
            self.infos_gt = pkl.load(info_file)
        with open(f"infos/{dataset_name}_short.info", "rb") as info_file:
            self.infos_short = pkl.load(info_file)
        self.data_info = self.infos_gt
        for i in range(len(self.data_info)):
            self.data_info[i]["hr"] = self.data_info[i]["data"]
            self.data_info[i]["lr"] = {dgain: self.infos_short[dgain][i] for dgain in self.infos_short}
            del self.data_info[i]["data"]

        self.id_remap = self.data_split()

    def __len__(self):
        return len(self.id_remap) * len(self.ratio_list)

    def data_split(self, eval_ids=None):
        id_remap = list(range(len(self.data_info)))
        if self.dataset_name == "indoor_x5":
            eval_ids = [4, 14, 25, 41, 44, 51, 52, 53, 58]
        elif self.dataset_name == "outdoor_x3":
            eval_ids = [9, 21, 22, 32, 44, 51]
        else:
            eval_ids = []

        id_remap = eval_ids
        return id_remap

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def hot_check(self, idx):
        if self.dataset_name == "indoor_x5":
            hot_ids = [6, 15, 33, 35, 39, 46, 37, 59]
        elif self.dataset_name == "outdoor_x3":
            hot_ids = [0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 26, 30, 51, 52, 54, 55, 56]
        else:
            raise NotImplementedError
        return True if idx in hot_ids else False

    def get_bias(self, iso=6400, exp=30, hot=False):
        if hot:
            bias = self.blc_mean_hot[iso][:, 0] * exp + self.blc_mean_hot[iso][:, 1]  # RGGB: (4,)
        else:
            bias = self.blc_mean[iso][:, 0] * exp + self.blc_mean[iso][:, 1]  # RGGB: (4,)
        return bias

    def blc_rggb(self, raw, bias):
        def _bayer2rggb(bayer):
            H, W = bayer.shape
            return bayer.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(H // 2, W // 2, 4)

        def _rggb2bayer(rggb):
            H, W, _ = rggb.shape
            return rggb.reshape(H, W, 2, 2).transpose(0, 2, 1, 3).reshape(H * 2, W * 2)

        return _rggb2bayer(_bayer2rggb(raw) + bias.reshape(1, 1, 4))

    def get_darkshading(self, iso=6400, hot=False):
        if iso not in self.darkshading:
            self.darkshading[iso] = np.load(f"./resources/IMX686/ds_{iso}.npy")
            self.darkshading_hot[iso] = np.load(f"./resources/IMX686/ds_{iso}_hot.npy")
        
        ds = self.darkshading_hot[iso] if hot else self.darkshading[iso]
        return ds

    def __getitem__(self, idx):
        dgain = self.ratio_list[idx // len(self.id_remap)]

        idr = self.id_remap[idx % len(self.id_remap)]
        hr_raw = np.load(self.data_info[idr]["hr"])
        lr_id = 0
        lr_raw = np.array(rawpy.imread(self.data_info[idr]["lr"][dgain]["data"][lr_id]).raw_image_visible)
        hr_raw, lr_raw = hr_raw.astype(np.float32), lr_raw.astype(np.float32)

        ## subtract dark shading
        lr_raw = lr_raw - self.get_darkshading(
            iso=self.iso,
            hot=self.hot_check(int(self.data_info[idr]["name"][-3:])),
        )

        ## pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [h, w, c]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [h, w, c]

        ## augmentation and crop to patches
        lr_crops = torch.FloatTensor(lr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        hr_crops = torch.FloatTensor(hr_imgs).unsqueeze(0).permute(0, 3, 1, 2)

        lr_crops *= dgain

        data = {
            "name": f"{self.data_info[idr]['name']}_x{dgain:02d}",
            "wb": torch.FloatTensor(self.data_info[idr]["wb"]),
            "ccm": torch.FloatTensor(self.data_info[idr]["ccm"]),
            "iso": torch.ones(hr_crops.shape[0]) * self.iso,
            "rgb_gain": torch.ones(hr_crops.shape[0]),
            "ratio": torch.ones(hr_crops.shape[0]) * dgain,
            "lr": torch.clamp(lr_crops, self.clip_low, self.clip_high),
            "hr": torch.clamp(hr_crops, 0, 1),
        }

        return data
