# Multilingual Summary Generation


The baselines that were used are located in the `baselines` folder. Their respective results along with the results from our model are in the `results` folder.

In the `Models` folder you will find all the necessary files to sample (i.e. beam-sample) from our trained models in both Arabic and Esperanto.

1. First you need to run the shell script located at: `Models/download_trained_models.sh` in order to download all the required trained models and post-processing files.
2. After you follow the instructions and install [Torch](http://torch.ch/) on your machine, run `Models/beam-sample.lua`.
3. Make sure that the following Python packages are installed: (i) `h5py`, (ii) `pandas`, and (iii) <`numpy` are installed.
3. Run `Models/beam-sample.py` in order to create a `.csv` file with the sampled summaries.

Please bare in mind that you need to have access to a GPU with at least 8 GB of memory in order to sample from the trained models. The experiments were conducted on a Titan X (Pascal) GPU. For all the possible sampling alterations, please check the comment sections of the above mentioned files.
