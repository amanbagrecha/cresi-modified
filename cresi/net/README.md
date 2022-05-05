Deep learning segmentation model, adapted from: https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution


```
root@2f94ec1f9a8d:/opt/cresi/cresi# python 02_eval.py configs/dar_tutorial_cpu.json
Executing inference on the CPU
Run utils.update_config()...
Updated config: {'path_src': '/opt/cresi/cresi', 'path_results_root': '/opt/cresi/results', 'save_weights_dir': '/opt/cresi/results/aws_weights/weights', 'num_channels': 3, 'network': 'resnet34', 'skeleton_thresh': 0.25, 'use_medial_axis': 0, 'min_subgraph_length_pix': 600, 'min_spur_length_m': 12, 'rdp_epsilon': 1, 'skeleton_band': 7, 'intersection_band': -1, 'early_stopper_patience': 5, 'num_folds': 1, 'default_val_perc': 0.2, 'train_data_refined_dir_ims': '', 'train_data_refined_dir_masks': '', 'speed_conversion_file': '/opt/cresi/cresi/configs/speed_conversion_binned7.csv', 'folds_file_name': 'folds4.csv', 'test_data_refined_dir': '/opt/cresi/test_imagery/dar/PS-RGB_8bit_clip', 'test_results_dir': 'dar_tutorial_cpu', 'test_sliced_dir': '/opt/cresi/test_imagery/dar/PS-RGB_8bit_clip_sliced', 'slice_x': 1300, 'slice_y': 1300, 'stride_x': 1280, 'stride_y': 1280, 'GSD': 0.3, 'tile_df_csv': 'tile_df.csv', 'folds_save_dir': 'folds', 'merged_dir': 'merged', 'stitched_dir_raw': 'stitched/mask_raw', 'stitched_dir_norm': 'stitched/mask_norm', 'stitched_dir_count': 'stitched/mask_count', 'wkt_submission': 'wkt_submission_nospeed.csv', 'skeleton_dir': 'skeleton', 'skeleton_pkl_dir': 'sknw_gpickle', 'graph_dir': 'graphs', 'iter_size': 1, 'target_rows': 1344, 'target_cols': 1344, 'loss': {'soft_dice': 0.25, 'focal': 0.75}, 'optimizer': 'adam', 'lr': 0.00015, 'lr_steps': [20, 25], 'lr_gamma': 0.2, 'batch_size': 12, 'epoch_size': 8, 'nb_epoch': 70, 'predict_batch_size': 1, 'test_pad': 64, 'num_classes': 8, 'warmup': 0, 'ignore_target_size': False, 'padding': 22, 'eval_rows': 1344, 'eval_cols': 1344, 'log_to_console': 0}
(len(config.test_sliced_dir) > 0) and (config.slice_x > 0), executing tile_im.py..
slice command: python /opt/cresi/cresi/data_prep/tile_im.py configs/dar_tutorial_cpu.json
Output path for sliced images: /opt/cresi/test_imagery/dar/PS-RGB_8bit_clip_sliced
Slicing images in: /opt/cresi/test_imagery/dar/PS-RGB_8bit_clip
im_path: /opt/cresi/test_imagery/dar/PS-RGB_8bit_clip/AOI_10_Dar_Es_Salaam_PS-MS_COG_clip.tif
im.shape: (11111, 11770, 3)
n pixels: 130776470
count: 50 x: 6400 y: 5120
  len df; 90
  Time to slice arrays: 58.59606313705444 seconds
  Total pixels in test image(s): 130776470
df saved to file: /opt/cresi/results/dar_tutorial_cpu/tile_df.csv
log_file: /opt/cresi/results/dar_tutorial_cpu/test.log
save_dir: /opt/cresi/results/dar_tutorial_cpu/folds
paths: {'masks': '', 'images': '/opt/cresi/test_imagery/dar/PS-RGB_8bit_clip_sliced'}
fn_mapping: {'masks': <function <lambda> at 0x7f2330611b80>}
image_suffix:
num_workers: 0
fold: 0
run eval.Evaluator.predict()...
prefix: fold0_
Creating datasets within pytorch_utils/eval.py()...
len val_dl: 90
self.num_workers 12
Running eval.read_model()...
load model with cpu
  model: Resnet34_upsample(
  (bottlenecks): ModuleList(
    (0): ConvBottleneck(
      (seq): Sequential(
        (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
      )
    )
    (1): ConvBottleneck(
      (seq): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
      )
    )
    (2): ConvBottleneck(
      (seq): Sequential(
        (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
      )
    )
    (3): ConvBottleneck(
      (seq): Sequential(
        (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
      )
    )
  )
  (decoder_stages): ModuleList(
    (0): UnetDecoderBlock(
      (layer): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): ReLU(inplace=True)
      )
    )
    (1): UnetDecoderBlock(
      (layer): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): ReLU(inplace=True)
      )
    )
    (2): UnetDecoderBlock(
      (layer): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): ReLU(inplace=True)
      )
    )
    (3): UnetDecoderBlock(
      (layer): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): ReLU(inplace=True)
      )
    )
  )
  (last_upsample): UnetDecoderBlock(
    (layer): Sequential(
      (0): Upsample(scale_factor=2.0, mode=nearest)
      (1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (2): ReLU(inplace=True)
    )
  )
  (final): Sequential(
    (0): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (encoder_stages): ModuleList(
    (0): Sequential(
      (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (1): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (4): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (5): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
)
  model sucessfully loaded
  0%|                                                                            | 0/90 [00:00<?, ?it/s]  eval.py -  - Evaluator - predict() - len samples: 1
  eval.py - Evaluator - predict()- samples.shape: torch.Size([1, 3, 1344, 1344])
  eval.py - Evaluator - predict() - predicted.shape: (1, 1344, 1344, 8)
  eval.py - Evaluator - predict() - data['image'].shape: torch.Size([1, 3, 1344, 1344])
  1%|▋                                                                 | 1/90 [00:40<1:00:17, 40.65s/it]  eval.py -  - Evaluator - predict() - len samples: 1
  eval.py - Evaluator - predict()- samples.shape: torch.Size([1, 3, 1344, 1344])
  eval.py - Evaluator - predict() - predicted.shape: (1, 1344, 1344, 8)
  eval.py - Evaluator - predict() - data['image'].shape: torch.Size([1, 3, 1344, 1344])
  2%|█▌                                                                  | 2/90 [01:16<55:31, 37.86s/it]  eval.py -  - Evaluator - predict() - len samples: 1
  eval.py - Evaluator - predict()- samples.shape: torch.Size([1, 3, 1344, 1344])
  eval.py - Evaluator - predict() - predicted.shape: (1, 1344, 1344, 8)
  eval.py - Evaluator - predict() - data['image'].shape: torch.Size([1, 3, 1344, 1344])
  3%|██▎                                                                 | 3/90 [01:50<52:34, 36.26s/it]  eval.py -  - Evaluator - predict() - len samples: 1
  eval.py - Evaluator - predict()- samples.shape: torch.Size([1, 3, 1344, 1344])
  eval.py - Evaluator - predict() - predicted.shape: (1, 1344, 1344, 8)

  ```