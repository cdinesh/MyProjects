Canonical documentation for this project is in **[README.md](./README.md)** (portfolio + notebook inventory).

The sections below are retained for **course environment** setup (Georgia Tech ISyE-style clone paths). Prefer the paths on your machine when working locally.

---

# Steps to run the analysis

# 1. Clone the Repository

git clone https://github.gatech.edu/ISyE6414-SP2025/quiet_titans.git
cd dchoudhari3

# 2. Create a Python Virtual Environment
# Create virtual environment to avoid dependency conflicts
python3 -m venv venv54
source venv54/bin/activate  # On Windows: venv54\Scripts\activate

# if import xgboost as xg error 
# XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
# Likely causes:
#   * OpenMP runtime is not installed (vcomp140.dll or libgomp-1.dll for Windows, 
#       libomp.dylib for Mac OSX, libgomp.so for Linux and other UNIX-like OSes). 
#       Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
#  * You are running 32-bit Python on a 64-bit OS
# then perform following steps
# Mac OS users. 
# install homebrew if not installed. This is needed for XGBoost import into jupyter notebook

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Set path /opt/homebrew/bin 
echo >> /Users/dinesh/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/your username/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Then install OpenMP runtime for MacOS Users. 
brew install libomp

# 3. Install Required Packages
# Install all dependencies using the provided `requirement.txt` file:
pip install -r requirement.txt

# 4. Launch Jupyter Notebook
# Install Jupyter if not installed
pip install notebook

# Start Jupyter:
jupyter notebook

# 5. Open and Run the Notebook
# In the Jupyter interface, navigate to `python/Group54_analysis.ipynb`.
# Open the notebook.
# Run each cell in order (use "Kernel > Restart & Run All" for a fresh run).


**Note:**  
1. All required data files are in the `data/` folder. Ensure this folder is present in the same directory as the notebook.
2. The notebook will generate summary statistics, visualizations, and model results.
3. Visualizations generated from notebook are saved in `image/` folder
4. The processed dataset will be saved as `data/final_dataset.csv`.
5. Analysis saves figure in `images/` folder

**Execution Time**
1. Preprocessing step to merge files: Approx: 15s
2. Combining and creating final_dataset.csv: 12s
3. All analysis steps: < 1s

**Optional Note for Mac Users**
If jupyter notebook gives XGBoostError while running notebook, perform following,
1. Install OpenMP using homebrew
brew install libomp
2. Restart your jupyter notebook kernel after install
3. If the error persists, try reinstalling XGBoost
pip uninstall xgboost
pip install xgboost 
4. On Apple Silicon 
pip install xgboost --no-binary xgboost

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
