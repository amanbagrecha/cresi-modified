import os
import subprocess
from  data_prep import create_8bit_images

# Create Directories
cresi_dir = '/opt/cresi'
src_dir = os.path.join(cresi_dir, 'cresi')
config_dir = os.path.join(cresi_dir, 'cresi/configs')
weight_dir = os.path.join(cresi_dir, 'results/aws_weights')
test_im_raw_dir = os.path.join(cresi_dir, 'test_imagery/dar/PS-MS')
test_im_clip_dir = os.path.join(cresi_dir, 'test_imagery/dar/PS-MS_clip')
test_final_dir = os.path.join(cresi_dir, 'test_imagery/dar/PS-RGB_8bit_clip')
results_root_dir = os.path.join(cresi_dir, 'results')
results_dir = os.path.join(results_root_dir, 'dar_tutorial_cpu')
mask_pred_dir = os.path.join(results_dir, 'folds')
mask_stitched_dir = os.path.join(results_dir, 'stitched/mask_norm')
# make dirs
for d in [weight_dir, test_im_raw_dir, test_im_clip_dir, test_final_dir]:
    os.makedirs(d, exist_ok=True)

# Clip the image extent
ulx, uly, lrx, lry = 39.25252, -6.7580, 39.28430, -6.7880  # v0

im_name = [z for z in os.listdir(test_im_raw_dir) if z.endswith('.tif')][0] # 30 cm raster
print("im_name:", im_name)
test_im_raw = os.path.join(test_im_raw_dir, im_name)
test_im_tmp = os.path.join(test_im_clip_dir, im_name.split('.tif')[0] + '_clip.vrt')
test_im_clip = os.path.join(test_im_clip_dir, im_name.split('.tif')[0] + '_clip_60cm.tif')


# clip to extent
args = f'gdal_translate -projwin {ulx} {uly} {lrx} {lry} {test_im_raw} {test_im_tmp}'
if os.path.exists(test_im_tmp):
    print(f"File exists, skipping!{test_im_tmp}")
else:
    subprocess.call(args, shell=True)
    print("temp file:", test_im_tmp)

# resample 30 cm imagery to 60 cm
args2 = f'gdal_translate -outsize 50% 50% {test_im_tmp} {test_im_clip}'
if os.path.exists(test_im_clip):
    print(f"File exists, skipping! {test_im_clip}")
else:
    subprocess.call(args2, shell=True)
    print("output_file:", test_im_clip)

print('#'*50)
print("Running create_8bit_images.py")
create_8bit_images.dir_to_8bit(test_im_clip_dir, test_final_dir,
                              command_file_loc='',
                              rescale_type="perc",
                              percentiles=[2,98],
                              band_order=[]) # specify [5,3,2] if MS channels. Here we are using RGB itself