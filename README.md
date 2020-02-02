# INSTALLATION #
conda create -n py37 python=3.7\
conda activate py37\
while read requirement; do conda install --yes $requirement; done < requirements.txt
