# Environment

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- Create Anaconda Environment:
  ```bash
  conda create -n RSSDFL python=3.7 ipykernel
  conda activate RSSDFL
  python -m ipykernel install --user --name RSSDFL --display-name "RSSDFL"
  ```

- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:
  ```bash
  conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
  ```

- Install other requirements:
  ```bash
  pip install -r requirements.txt
  ```


# Usage

Run demo.ipynb

(As our checkpoint exceeds the required 50 Mb maximum file size, we temporally remove it.
Results are shown in demo.ipynb.)