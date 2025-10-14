Updated: June 11th, 2025

For MAC M1 install (NO GPU acceleration)

# STEP 1: Create environment
conda create -n omnipose_working python=3.10.12 -y
conda activate omnipose_working

# STEP 2: Install STABLE PyTorch (not 2.7.0)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# STEP 3: Install core dependencies
pip install numpy==1.26.4 scipy==1.10.1 scikit-image==0.24.0 aicsimageio torch_optimizer

# STEP 4: Install STABLE Omnipose from PyPI
pip install omnipose

# STEP 5: Install additional dependencies
pip install napari pyqt5 pyqt6 mahotas opencv-python-headless
pip install natsort pyqtgraph omnipose-theme superqt darkdetect seaborn

# STEP 6: Test
omnipose --help