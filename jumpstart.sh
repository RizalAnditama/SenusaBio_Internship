
#!/bin/bash

wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 2. Install Miniconda with automated responses
echo -e "yes

yes" | bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# 3. Initialize Conda for the current shell session
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash

# 4. Accept Anaconda Terms of Service to bypass the error
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 5. Create Magpie Environment 
echo -e "a
a" | conda env create -f magpie.yml
conda run -n magpie pip install -r requirements.txt

# 6. Create SpliceAI Environment
conda env create -f spliceai.yml

echo "Jumpstart configuration completed successfully!"
