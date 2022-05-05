# Modified City-scale Road Extraction from Satellite Imagery (CRESI) 

This repository is forked from `https://github.com/CosmiQ/cresi`

- The upstream repository is older in ubuntu version (docker) and does not specify package version with the installation, causing dependency issues. This repo solves that issue. Currently cpu-only version is in the `main` branch.


____
## Install

1. Download/clone this repository `git clone git@github:amanbagrecha/cresi-modified`

2. pull docker image

		docker pull amanbagrecha/cresi:v1
	
3. Create docker container (all commands should be run in this container)

		docker run -it --rm -v /local/dir:/opt/cresi -p 9111:9111 --ipc=host --name cresi_cpu amanbagrecha/cresi:v1
____

## Test

An example test image can be found from the (Spacenet 5 challenge)[https://registry.opendata.aws/spacenet/].


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
# Download test data
aws s3 cp --recursive s3://spacenet-dataset/AOIs/AOI_10_Dar_Es_Salaam/PS-MS $weight_dir

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


