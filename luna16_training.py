import os
import argparse
import gc
import json
import logging
import sys
import time
import numpy as np
import torch
from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
    pad2factor
)
from torch.utils.tensorboard import SummaryWriter
from visualize_image import visualize_one_xy_slice_in_3d_image
from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from networks.retinanet_detector import RetinaNetDetector
from networks.retinanet_network import (
    RetinaNet,
    fpn_feature_extractor,
)
from networks.swin_unetr.swin_unetr  import SwinUNETR


from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, Dataset, box_utils, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.transforms import ScaleIntensityRanged
from monai.utils import set_determinism
import setproctitle
import warnings


warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'
setproctitle.setproctitle("detection")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_luna16_fold9.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_luna16_16g.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    set_determinism(seed=0)

    amp = True

    monai.config.print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    # 1. define transform
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=0,
        a_max=255.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    train_transforms = generate_detection_train_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        args.patch_size,
        batch_size=args.batch_size_per_image, # batch size per image not the input batch size
        affine_lps_to_ras=True,
        amp=amp,
    )

    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
    )

    # 2. prepare training data
    # create a training data loader
    train_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=args.data_base_dir,
    )
    train_ds = Dataset(
        data=train_data[: int(0.95 * len(train_data))],
        transform=train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )

    # create a validation data loader
    val_ds = Dataset(
        data=train_data[int(0.95 * len(train_data)) :],
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )

    # 3. build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) build anchor generator
    # returned_layers: when target boxes are small, set it smaller
    # base_anchor_shapes: anchor shape for the most high-resolution output,
    # when target boxes are small, set it smaller
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(args.returned_layers) + 1)],
        base_anchor_shapes=args.base_anchor_shapes,
    )

    if not args.resume_checkpoint:
        # 2) build network

        # Swin-UNETR
        backbone = SwinUNETR(
            img_size=(64, 64, 64),
            in_channels=1,
            out_channels=128,
            depths=(2, 2, 4, 6),
            num_heads=(3, 6, 12, 24),
            feature_size=48,
            norm_name="instance",
            drop_rate=0.1,
            attn_drop_rate=0.1,
            dropout_path_rate=0.1,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
            downsample="merging",
            block_inplanes=args.block_inplanes
        )
   

        feature_extractor = fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=args.spatial_dims,
        )
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        size_divisible = [s * 2 * 2 ** max(args.returned_layers) for s in args.conv1_t_stride]

        net = RetinaNet(
            spatial_dims=args.spatial_dims,
            num_classes=len(args.fg_labels),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
            train_parallel=args.train_parallel,
        )

        # 3) build detector
        detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)

        # set training components
        detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
        detector.set_hard_negative_sampler(
            batch_size_per_image=64,
            positive_fraction=args.balanced_sampler_pos_fraction,
            pool_size=20,
            min_neg=16,
        )
        detector.set_target_keys(box_key="box", label_key="label")

        # set validation components
        detector.set_box_selector_parameters(
            score_thresh=args.score_thresh,
            topk_candidates_per_level=1000,
            nms_thresh=args.nms_thresh,
            detections_per_img=100,
        )
        detector.set_sliding_window_inferer(
            roi_size=args.val_patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )

        # 4. Initialize training
        # initlize optimizer
        optimizer = torch.optim.SGD(
            detector.network.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=3e-5,
            nesterov=True,
        )
        after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler)
        scaler = torch.cuda.amp.GradScaler() if amp else None
        optimizer.zero_grad()
        optimizer.step()

    # load model from checkpoint
    else:
        net = torch.load(env_dict["model_path"]).to(device)
        print(f"Load model from {env_dict['model_path']}")

        # 3) build detector
        detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)

        # set training components
        detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
        detector.set_hard_negative_sampler(
            batch_size_per_image=64,
            positive_fraction=args.balanced_sampler_pos_fraction,
            pool_size=20,
            min_neg=16,
        )
        detector.set_target_keys(box_key="box", label_key="label")

        # set validation components
        detector.set_box_selector_parameters(
            score_thresh=args.score_thresh,
            topk_candidates_per_level=1000,
            nms_thresh=args.nms_thresh,
            detections_per_img=100,
        )
        detector.set_sliding_window_inferer(
            roi_size=args.val_patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )

        # 4. Initialize training
        # initlize optimizer
        optimizer = torch.optim.SGD(
            detector.network.parameters(),
            0.001,
            momentum=0.9,
            weight_decay=3e-5,
            nesterov=True,
        )
        scaler = torch.cuda.amp.GradScaler() if amp else None
        optimizer.zero_grad()
        optimizer.step()

    # initialize tensorboard writer
    tensorboard_writer = SummaryWriter(args.tfevent_path)

    # 5. train
    val_interval = 10  # do validation every val_interval epochs
    coco_metric = COCOMetric(classes=["nodule"], iou_list=[0.1], max_detection=[100])
    best_val_epoch_metric = 0.0
    best_val_epoch = -1  # the epoch that gives best validation metrics

    max_epochs = 200
    epoch_len = len(train_ds) // train_loader.batch_size
    w_cls = config_dict.get("w_cls", 1.0)  # weight of classification loss, default 1.0
    w_reg = config_dict.get("w_reg", 1.0)  # weight of box regression loss, default 1.0
    for epoch in range(max_epochs):
        # ------------- Training -------------
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        detector.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_reg_loss = 0
        step = 0
        start_time = time.time()
        if not args.resume_checkpoint:
            scheduler_warmup.step()
        # Training
        for batch_data in train_loader:
            step += 1
            inputs = [
                batch_data_ii["image"].to(device) for batch_data_i in batch_data for batch_data_ii in batch_data_i
            ]
            targets = [
                dict(
                    label=batch_data_ii["label"].to(device),
                    box=batch_data_ii["box"].to(device),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            for param in detector.network.parameters():
                param.grad = None

            if amp and (scaler is not None):
                with torch.cuda.amp.autocast():
                    outputs = detector(inputs, targets)
                    loss = w_cls * outputs[detector.cls_key] + w_reg * outputs[detector.box_reg_key] 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = detector(inputs, targets)
                loss = w_cls * outputs[detector.cls_key] + w_reg * outputs[detector.box_reg_key]
                loss.backward()
                optimizer.step()

            # save to tensorboard
            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[detector.cls_key].detach().item()
            epoch_box_reg_loss += outputs[detector.box_reg_key].detach().item()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            tensorboard_writer.add_scalar("train_loss", loss.detach().item(), epoch_len * epoch + step)

        end_time = time.time()
        print(f"Training time: {end_time-start_time}s")
        del inputs, batch_data
        torch.cuda.empty_cache()
        gc.collect()

        # save to tensorboard
        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_box_reg_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        tensorboard_writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_cls_loss", epoch_cls_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_box_reg_loss", epoch_box_reg_loss, epoch + 1)
        tensorboard_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch + 1)
    
        # save last trained model
        torch.save(detector.network, env_dict["model_path"][:-3] + "_last.pt")
        print("saved last model")

        # ------------- Validation for model selection -------------
        if (epoch + 1) % val_interval == 0:
            detector.eval()
            val_outputs_all = []
            val_targets_all = []
            start_time = time.time()
            with torch.no_grad():
                for val_data in val_loader:
                 
                    val_inputs = [pad2factor(val_data_i.pop("image"), factor=64).to(device) for val_data_i in val_data]

                    if amp:
                        with torch.cuda.amp.autocast():
                            val_outputs = detector(val_inputs, use_inferer=True)
                    else:
                        val_outputs = detector(val_inputs, use_inferer=True)

                    # save outputs for evaluation
                    val_outputs_all += val_outputs
                    val_targets_all += val_data

            end_time = time.time()
            print(f"Validation time: {end_time-start_time}s")

            # visualize an inference image and boxes to tensorboard
            draw_img = visualize_one_xy_slice_in_3d_image(
                gt_boxes=val_data[0]["box"].cpu().detach().numpy(),
                image=val_inputs[0][0, ...].cpu().detach().numpy(),
                pred_boxes=val_outputs[0][detector.target_box_key].cpu().detach().numpy(),
            )
            tensorboard_writer.add_image("val_img_xy", draw_img.transpose([2, 1, 0]), epoch + 1)

            # compute metrics
            del val_inputs
            torch.cuda.empty_cache()
            results_metric = matching_batch(
                iou_fn=box_utils.box_iou,
                iou_thresholds=coco_metric.iou_thresholds,
                pred_boxes=[
                    val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_scores=[
                    val_data_i[detector.pred_score_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                gt_boxes=[val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_targets_all],
                gt_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_targets_all
                ],
            )
            val_epoch_metric_dict = coco_metric(results_metric)[0]
            print(val_epoch_metric_dict)

            # write to tensorboard event
            for k in val_epoch_metric_dict.keys():
                tensorboard_writer.add_scalar("val_" + k, val_epoch_metric_dict[k], epoch + 1)
            val_epoch_metric = val_epoch_metric_dict.values()
            val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
            tensorboard_writer.add_scalar("val_metric", val_epoch_metric, epoch + 1)

            # save best trained model
            if val_epoch_metric > best_val_epoch_metric:
                best_val_epoch_metric = val_epoch_metric
                best_val_epoch = epoch + 1
                torch.save(detector.network, env_dict["model_path"])
                print("saved new best metric model")
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_metric, best_val_epoch_metric, best_val_epoch
                )
            )

    print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
    tensorboard_writer.close()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
