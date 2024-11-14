import sys
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig
from utils import draw

from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *


warnings.simplefilter(action="ignore", category=FutureWarning)


def do_test(cfg, model, use_dark_inference=False):
    val_loader = instantiate(cfg.dataloader.val)

    i = 0

    model.train(False)
    AUC = []
    dist = []
    inout_gt = []
    inout_pred = []
    with torch.no_grad():
        for data in val_loader:
            val_gaze_heatmap_pred, val_gaze_inout_pred = model(data)
            val_gaze_heatmap_pred = (
                val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
            )
            val_gaze_inout_pred = val_gaze_inout_pred.cpu().detach().numpy()

            # go through each data point and record AUC, dist, ap
            for b_i in range(len(val_gaze_heatmap_pred)):
                auc_batch = []
                dist_batch = []
                if data["gaze_inouts"][b_i]:
                    # remove padding and recover valid ground truth points
                    valid_gaze = data["gazes"][b_i]
                    # AUC: area under curve of ROC
                    multi_hot = data["heatmaps"][b_i]
                    multi_hot = (multi_hot > 0).float().numpy()
                    if use_dark_inference:
                        pred_x, pred_y = dark_inference(val_gaze_heatmap_pred[b_i])
                    else:
                        pred_x, pred_y = argmax_pts(val_gaze_heatmap_pred[b_i])
                    norm_p = [
                        pred_x / val_gaze_heatmap_pred[b_i].shape[-1],
                        pred_y / val_gaze_heatmap_pred[b_i].shape[-2],
                    ]
                    scaled_heatmap = np.array(
                        Image.fromarray(val_gaze_heatmap_pred[b_i]).resize(
                            (64, 64),
                            resample=Image.Resampling.BILINEAR,
                        )
                    )
                    auc_score = auc(scaled_heatmap, multi_hot)
                    auc_batch.append(auc_score)
                    dist_batch.append(L2_dist(valid_gaze.numpy(), norm_p))
                    
                    save_image = True

                    # Save the image with prediction if flag is True
                    if save_image:
                        # print(data.keys())
                        # pil_image = to_pil_image(data["images"][b_i])
                        # print('data["images"][b_i]', data["images"][b_i].shape)
                        # image = np.array(pil_image)
                        # h, w, _ = image.shape
                        # cv2.circle(image, (int(pred_x * w), int(pred_y * h)), 5, (0, 255, 0), -1)
                        # cv2.imwrite('output.png', image)
                        draw(data, heatmap=scaled_heatmap, out_path=f'output/output{i}.png')
                        i += 1
                        if i == 10:
                            exit()
                        break
                            
                AUC.extend(auc_batch)
                dist.extend(dist_batch)
                break
            inout_gt.extend(data["gaze_inouts"].cpu().numpy())
            inout_pred.extend(val_gaze_inout_pred)

    print("|AUC   |dist    |AP     |")
    print(
        "|{:.4f}|{:.4f}  |{:.4f}  |".format(
            torch.mean(torch.tensor(AUC)),
            torch.mean(torch.tensor(dist)),
            ap(inout_gt, inout_pred),
        )
    )

    with open('output.txt', 'w') as f:
        f.write("|AUC   |dist    |AP     |\n")
        f.write(
            "|{:.4f}|{:.4f}  |{:.4f}  |\n".format(
                torch.mean(torch.tensor(AUC)),
                torch.mean(torch.tensor(dist)),
                ap(inout_gt, inout_pred),
            )
        )


def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights)["model"])
    model.to(cfg.train.device)
    do_test(cfg, model, use_dark_inference=args.use_dark_inference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    parser.add_argument(
        "--model_weights",
        type=str,
        help="model weights",
    )
    parser.add_argument("--use_dark_inference", action="store_true")
    args = parser.parse_args()
    main(args)
