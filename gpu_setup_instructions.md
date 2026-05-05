# Switch to Python 3.11 and Install GPU Stack

Since Python 3.14 is incompatible with current CUDA wheels, we will use the installed Python 3.11.

## Steps

1.  Create a virtual environment using Python 3.11:
    ```powershell
    py -3.11 -m venv venv_gpu
    ```
2.  Activate the environment:
    ```powershell
    .\venv_gpu\Scripts\Activate.ps1
    ```
3.  Install PyTorch with CUDA 12.1:
    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
4.  Install dependencies:
    ```powershell
    pip install pandas numpy scikit-learn matplotlib seaborn librosa
    ```
5.  Run training:
    ```powershell
    python train_baseline.py --model cnn
    ```
