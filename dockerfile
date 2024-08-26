# Use a base image with Python and Conda
FROM continuumio/miniconda3

# Install Neovim and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    neovim \
    curl && \
    apt-get clean

# Clone NvChad configuration for Neovim
# RUN git clone https://github.com/NvChad/starter ~/.config/nvim

# Set up Conda environment and update Conda itself
RUN conda update -n base -c defaults conda -y && \
    conda config --add channels conda-forge && \
    conda create -n ml_env python=3.9 -y && \
    conda install -n ml_env -y \
        numpy \
        pandas \
        scikit-learn \
        matplotlib \
        seaborn \
        scipy \
        jupyter \
        opencv \
        pillow \
        nltk \
        spacy && \
    conda clean -afy

# Set the default shell to activate the environment
RUN echo "conda activate ml_env" >> ~/.bashrc

VOLUME /data

# Set the working directory
WORKDIR /data

# Run a command (optional, for example, starting a Jupyter notebook)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]

# Expose the port for Jupyter notebook
EXPOSE 8888

