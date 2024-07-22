# create and activate conda environment
conda update -n base -c defaults conda
conda create -p ./venv python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate ./venv

# install conda packages
conda install -c conda-forge pynini -y
conda install 'ffmpeg<4.4' -y

# install pip dependencies
pip install --no-input --upgrade pip
pip install --no-input Cython==0.29.34
pip install --no-input -r ./requirements.txt
