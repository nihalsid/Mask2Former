# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import gzip

# fmt: off
import sys

from matplotlib import cm

from tta_handler import TTAHandler

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings
import torch
import cv2
import albumentations as A
import numpy as np
import tqdm
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from PIL import Image
import torchvision.transforms as T
import math

# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def visualize_tensor(tensor, minval=0.000, maxval=1.00, use_global_norm=True):
    x = tensor.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    if use_global_norm:
        mi = minval
        ma = maxval
    else:
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x_ = Image.fromarray((cm.get_cmap('jet')(x) * 255).astype(np.uint8))
    x_ = T.ToTensor()(x_)[:3, :, :]
    return x_


def probability_to_normalized_entropy(probabilities):
    entropy = torch.zeros_like(probabilities[:, :, 0])
    for i in range(probabilities.shape[2]):
        entropy = entropy - probabilities[:, :, i] * torch.log2(probabilities[:, :, i] + 1e-8)
    entropy = entropy / math.log2(probabilities.shape[2])
    return entropy


def load_and_save_with_entropy_and_confidence(out_filename, entropy, confidences):
    from torchvision.io import read_image
    from torchvision.utils import save_image
    org_img = read_image(out_filename).float() / 255.0
    e_img = visualize_tensor(1 - entropy)
    c_img = visualize_tensor(confidences)
    save_image(torch.cat([org_img.unsqueeze(0), e_img.unsqueeze(0), c_img.unsqueeze(0)], dim=0), out_filename, value_range=(0, 1), normalize=True)


def save_panoptic(predictions, predictions_notta, _demo, out_filename):
    mask, segments, probabilities, confidences = predictions["panoptic_seg"]
    mask_notta, segments_notta, _, confidences_notta = predictions_notta["panoptic_seg"]
    # since we use cat_ids from scannet, no need for mapping
    # for segment in segments:
    #     cat_id = segment["category_id"]
    #     segment["category_name"] = demo.metadata.stuff_classes[cat_id]
    with gzip.open(out_filename, "wb") as fid:
        torch.save(
            {
                "mask": mask,
                "segments": segments,
                "mask_notta": mask_notta,
                "segments_notta": segments_notta,
                "confidences_notta": confidences_notta,
                "probabilities": probabilities,
                "confidences": confidences,
                # "feats": predictions["res3_feats"]
            }, fid
        )


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--predictions",
        help="Save raw predictions together with visualizations."
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Max procs",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=0,
        help="Current proc",
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    if args.input:
        all_files = sorted(list(Path(args.input[0]).iterdir()))
        used_files = [x for x_i, x in enumerate(all_files) if x_i % args.n == args.p]
        used_files = [x for x in enumerate(used_files) if not Path(os.path.splitext(os.path.join(args.output, os.path.basename(str(x))))[0] + ".ptz").exists()]
        for path in tqdm.tqdm(used_files, disable=not args.output):
            path = str(path)
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            augmentations = [A.HorizontalFlip(always_apply=True), A.RGBShift(always_apply=True), A.CLAHE(always_apply=True), A.RandomGamma(always_apply=True, gamma_limit=(80, 120)), A.RandomBrightnessContrast(always_apply=True),
                             A.MedianBlur(blur_limit=7, always_apply=True), A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), always_apply=True)]
            augmentations.extend([A.Compose([augmentations[1], augmentations[2]]),
                                  A.Compose([augmentations[2], augmentations[3]]),
                                  A.Compose([augmentations[1], augmentations[3]]),
                                  A.Compose([augmentations[2], augmentations[4]]),
                                  A.Compose([augmentations[5], augmentations[6]])])
            use_old_augmentation_method = False
            if use_old_augmentation_method:
                predictions, visualized_output = demo.run_on_image(img)
                predictions['panoptic_seg'] = list(predictions['panoptic_seg'])
                averaged_probs, averaged_conf = predictions["panoptic_seg"][2].cpu(), predictions["panoptic_seg"][3].cpu()
                # averaged_feats = predictions['res3_feats'].cpu()
                for aud_idx, augmentation in enumerate(augmentations):
                    transformed_image = augmentation(image=img)["image"]
                    aug_pred, _ = demo.run_on_image(transformed_image, visualize=False)
                    if not aud_idx == 0:
                        aug_probs, aug_conf = aug_pred["panoptic_seg"][2], aug_pred["panoptic_seg"][3]
                        # aug_feat = aug_pred['res3_feats']
                    else:
                        aug_probs, aug_conf = torch.fliplr(aug_pred["panoptic_seg"][2]), torch.fliplr(aug_pred["panoptic_seg"][3])
                        # aug_feat = torch.fliplr(aug_pred['res3_feats'])
                    averaged_probs += aug_probs.cpu()
                    averaged_conf += aug_conf.cpu()
                    # averaged_feats += aug_feat.cpu()
                averaged_probs /= (len(augmentations) + 1)
                averaged_conf /= (len(augmentations) + 1)
                # averaged_feats /= (len(augmentations) + 1)
                predictions["panoptic_seg"][2], predictions["panoptic_seg"][3] = averaged_probs, averaged_conf
                # predictions['res3_feats'] = averaged_feats
            else:
                predictions_0, _ = demo.run_on_image(img, visualize=False)
                # averaged_feats = predictions_0['res3_feats'].cpu()
                list_aug_probs, list_aug_confs = [x.cpu() for x in predictions_0["panoptic_seg"][0]], [x.cpu() for x in predictions_0["panoptic_seg"][1]]
                for aud_idx, augmentation in enumerate(augmentations):
                    transformed_image = augmentation(image=img)["image"]
                    aug_pred, _ = demo.run_on_image(transformed_image, visualize=False)
                    if not aud_idx == 0:
                        aug_probs, aug_conf = aug_pred["panoptic_seg"][0], aug_pred["panoptic_seg"][1]
                        # aug_feat = aug_pred['res3_feats']
                    else:
                        aug_probs, aug_conf = aug_pred["panoptic_seg"][0], torch.fliplr(aug_pred["panoptic_seg"][1].permute((1, 2, 0))).permute((2, 0, 1))
                        # aug_feat = torch.fliplr(aug_pred['res3_feats'])
                    aug_probs = aug_probs.cpu()
                    aug_conf = aug_conf.cpu()
                    list_aug_probs.extend([x for x in aug_probs])
                    list_aug_confs.extend([x for x in aug_conf])
                    # averaged_feats += aug_feat.cpu()
                # averaged_feats /= (len(augmentations) + 1)
                tta_handler_start_time = time.time()
                tta_handler = TTAHandler(list_aug_probs, list_aug_confs)
                probabilities, confidences = tta_handler.find_tta_probabilities_and_masks()
                print(f'TTA Handler time: {time.time() - tta_handler_start_time:.2f}s')
                del tta_handler
                # todo: deleted visualizations for now, turn on if needed
                # predictions, visualized_output = demo.run_post_augmentation(img, probabilities, confidences, visualize=True)
                predictions, visualized_output = demo.run_post_augmentation(img, probabilities, confidences, visualize=False)
                predictions_no_tta, _ = demo.run_post_augmentation(img, predictions_0["panoptic_seg"][0], predictions_0["panoptic_seg"][1], visualize=False)
                # predictions['res3_feats'] = averaged_feats
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                if visualized_output is not None:
                    visualized_output.save(out_filename)
                    probabilities, confidences = predictions["panoptic_seg"][2], predictions["panoptic_seg"][3]
                    entropy = probability_to_normalized_entropy(probabilities)
                    load_and_save_with_entropy_and_confidence(out_filename, entropy, confidences)
                if args.predictions:
                    out_filename_noext, _ = os.path.splitext(out_filename)
                    save_panoptic(predictions, predictions_no_tta, demo, out_filename_noext + ".ptz")
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
