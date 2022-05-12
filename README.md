# Modified City-scale Road Extraction from Satellite Imagery (CRESI) 

This repository is forked from `https://github.com/CosmiQ/cresi`

- The upstream repository is older in ubuntu version (docker) and does not specify package version with the installation, causing dependency issues. This repo solves that issue. Currently cpu-only version is in the `main` branch.


____

## Install & Test


1. Download/clone this repository `git clone git@github:amanbagrecha/cresi-modified`

2. pull docker image

		docker pull amanbagrecha/cresi:v1
	
3. Create docker container (all commands should be run in this container)

		docker run -it --rm -v /local/dir:/opt/cresi -p 9111:9111 --ipc=host --name cresi_cpu amanbagrecha/cresi:v1
____
An example test image can be found from the [Spacenet 5 challenge](https://registry.opendata.aws/spacenet/).


Here we downsample the default GSD of the test image from 0.3 cm to 0.6 cm. More info about this below.

Inference can be perform either using shell script or using jupyter notebook.

#### 1. Via Jupyter notebook

Launch jupyter notebook by typing in the following command

```sh
cd /opt/cresi/notebooks/dar_tutorial_cpu/

jupyter notebook --ip 0.0.0.0 --port=9111 --no-browser --allow-root &
```

Open up your browser and type `http://localhost:9111/notebooks/`.
It would ask for token, you can copy from your terminal and paste it in browser.

Run the cell sequentially as described in [cresi_cpu_part2.ipynb](https://github.com/amanbagrecha/cresi-modified/blob/main/notebooks/dar_tutorial_cpu/cresi_cpu_part2.ipynb)

---

#### 2. Via shell script

```sh

# setup path
cresi_dir=/opt/cresi
src_dir=$cresi_dir/cresi
weight_dir=$cresi_dir/results/aws_weights
test_im_raw_dir=$cresi_dir/test_imagery/dar/PS-MS
# make dir if not exist
mkdir -p $cresi_dir $src_dir $weight_dir $test_im_raw_dir

# Download model weights
aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet-model-weights/spacenet-5/baseline/ $weight_dir

# Download test data
aws s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_10_Dar_Es_Salaam/PS-MS/ $test_im_raw_dir

# clip and downsample test data
python $src_dir/01b_prep.py

# perform inference
cd $src_dir
./test.sh configs/dar_tutorial_cpu.json
```	

#### 3. Run commands individually


	A. Execute inference (within docker image)

		python $src_dir/02_eval.py configs/dar_tutorial_cpu.json

	B. Merge predictions (if required)

		python $src_dir/03a_merge_preds.py configs/dar_tutorial_cpu.json
	
	C. Stitch together mask windows (if required)

		python $src_dir/03b_stitch.py configs/dar_tutorial_cpu.json

	D. Extract mask skeletons

		python $src_dir/04_skeletonize.py configs/dar_tutorial_cpu.json
	
	E. Create graph

		python $src_dir/05_wkt_to_G.py configs/dar_tutorial_cpu.json


# Install and Train (using GPU)

1. Download/clone this repository `git clone git@github:amanbagrecha/cresi-modified`

2. pull docker image

		docker pull amanbagrecha/cresi-gpu:v1
	
3. Create docker container (all commands should be run in this container) do not forget to change the `/local/dir`

		docker run -it --rm --gpus 1 -v /local/dir:/opt/cresi -p 9111:9111 --ipc=host --name cresi_gpu amanbagrecha/cresi-gpu:v1

Your `local/dir` should be the root of your git clone

```sh
# setup path
cresi_dir=/opt/cresi
src_dir=$cresi_dir/cresi
data_dir=$cresi_dir/data

# make dir if not exist
mkdir -p $cresi_dir $src_dir $data

# Download SN5 Roads for mumbai for training the model
aws s3 sync --no-sign-request s3://spacenet-dataset/spacenet/SN5_roads/train/AOI_8_Mumbai/PS-MS \
$data_dir/SN5_roads/AOI_8_Mumbai/PS-MS

# Download mumbai road geojson data
aws s3 sync --no-sign-request s3://spacenet-dataset/spacenet/SN5_roads/train/AOI_8_Mumbai/geojson_roads_speed \
$data_dir/SN5_roads/AOI_8_Mumbai/geojson_roads_speed

# downsample the data and convert to 8bit
python $src_dir/data_prep/create_8bit_images.py \
    --indir=$data_dir/SN5_roads/AOI_8_Mumbai/PS-MS \
    --outdir=$data_dir/cresi_data/train/8bit/PS-RGB \
    --rescale_type=perc \
    --percentiles=2,98 \
    --band_order=5,3,2

# create multichannel training masks
python $src_dir/data_prep/speed_masks.py \
  --geojson_dir=$data_dir/SN5_roads/AOI_8_Mumbai/geojson_roads_speed \
  --image_dir=$data_dir/SN5_roads/AOI_8_Mumbai/PS-MS \
  --output_conversion_csv_binned=$data_dir/cresi_data/train/SN5_roads_train_speed_conversion_binned.csv \
  --output_mask_dir=$data_dir/cresi_data/train/train_mask_binned \
  --output_mask_multidim_dir=$data_dir/cresi_data/train/train_mask_binned_mc \
  --buffer_distance_meters=2

# generate folds
python $src_dir/00_gen_folds.py $src_dir/config/sn5_baseline.json

# start training
python $src_dir/01_train.py $src_dir/config/sn5_baseline.json --fold=0


# Perform inference
# convert test image to 8bit
python $src_dir/data_prep/create_8bit_images.py \
    --indir=$data_dir/SN5_roads/AOI_8_Mumbai/PS-MS \
    --outdir=$data_dir/cresi_data/train/8bit/public_test/PS-RGB \
    --rescale_type=perc \
    --percentiles=2,98 \
    --band_order=5,3,2

# run eval
python $src_dir/02_eval.py $src_dir/config/sn5_baseline.json
```