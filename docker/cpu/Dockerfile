# # docker build -t cresi .
# # docker run -it --ipc=host -v /local_data:/opt/cresi -p 9111:9111 --name cresi_cpu cresi
# # To use jupyter notebook, inside container run:
# # jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=9111
# # back on local machine, invoke the following in a web browser: https://localhost:9111

#######################
# Adapted from: https://github.com/CosmiQ/cresi
FROM continuumio/conda-ci-linux-64-python3.8:latest

USER root

WORKDIR "/opt/cresi"


# use conda-forge instead of default channel
RUN conda update conda && \
    conda config --add channels conda-forge

# # install mamba for faster resolution
RUN conda install -y -c conda-forge mamba && \
    conda update mamba

ADD https://raw.githubusercontent.com/amanbagrecha/cresi-modified/main/requirements.yml ./

RUN mamba env update -f requirements.yml

RUN pip install torch==1.8.2+cpu torchvision==0.9.2+cpu \
-f https://download.pytorch.org/whl/lts/1.8/torch_lts.html \ 
&& pip install tensorboardX \
&& pip install utm \
&& pip install numba \
&& pip install torchsummary \
&& pip install imagecodecs

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini


RUN apt-get update && apt-get install libgl1 -y

# # # TensorBoard
EXPOSE 6006
# # # IPython
EXPOSE 9111

ENV PROJ_LIB /opt/conda/share/proj

RUN ["/bin/bash"]