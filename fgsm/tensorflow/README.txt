These instructions specify the steps to setup the environment, run the code in this project and reproduce results.
It also contains additional information on meta artifacts generated as part of experimentation.

Disclaimer: As part of this evaluation, all development was done on Paperspace Gradient, which provides an online
Jupyter Notebook instance with free GPUs for a maximum of 6 hours at a stretch. As part of available instances,
pre-configured Ubuntu image with TensorFlow v2.7 was provided with most basic Python packages such as NumPy, etc.
pre-installed. For more information, please check: https://gradient.run/notebooks

-----------------
Setup Environment
-----------------
However, if one were to setup the environment from scratch, please do the following (assuming Linux distribution with Nvidia GPU available):

1. Install a package manager like conda (Miniconda, preferred or Anaconda)
	(a) Miniconda installation instructions: https://docs.conda.io/en/latest/miniconda.html#linux-installers
	or, (but not both)
	(b) Anaconda installation instructions: https://docs.anaconda.com/anaconda/install/linux/

2. Create a virtual environment to avoid dependency clashes. If using conda, simply do:
		conda create -n [virtual-environment_name] python=[python_version]
   For more information, follow this: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python
   Subsequently, activate the virtual environment you created using "conda activate [virtual-environment_name]".

3. Install TensorFlow 2.7: https://www.tensorflow.org/install

4. Install necessary Python packages such as NumPy, Jupyter Notebook etc. using the conda package manager or pip package manager.
   The former often has a few variations and it is always wise to search on a web browser with "conda install [package_name]"
   prior to installation. For using pip, often "pip install [package_name]" is sufficient.

----------------------------
Run Code & Reproduce Results
----------------------------
There are primarily two ways to run the code in this project - using terminal or via IPython Notebooks. The latter is more elegant.

Note: As a first step, make sure that the dataset is available to the python script below. As experienced by the author, tensorflow
was unable to fetch the CIFAR-10 dataset from the server: https://www.cs.toronto.edu/~kriz/cifar.html. "Server Unavailable" error kept
popping up. As a workaround, the (python version) dataset was downloaded from the server and completely unzipped to get the "cifar10-batches-py"
directory in the same location as the python script below. Either of the approaches may then be followed to run the code and reproduce results.

1. Using terminal
   Make sure you are in the virtual environment you created and installed packages in. Simply run the "train_mixed_adversarial_defense.py"
   file in the current directory using "python train_mixed_adversarial_defense.py". All outputs will be printed to console.
   To just run the FGSM attack code, execute "python train_adversarial_attack.py" in the terminal.

2. For better viewability, use Jupyter Notebooks (make sure jupyter-notebook is installed)
   (a) Start Jupyter Notebook by typing "jupyter-notebook" in the virtual environment terminal.
   (b) Browse to "./ipynb/train_mixed_adversarial_defense.ipynb" (or "./ipynb/train_adversarial_attack.ipynb") and click to open the
	   notebook in a new tab.
   (c) Start execution of cells one after the other by hitting "Shift + Enter" on the keyboard. Certain cells may take longer
	   to execute owing to the nature of operation.

3. (Optional) - NOT VERIFIED
    To ensure packaging consistency and avoid host system dependencies, the project has been dockerised. Docker is a neat way to build,
	deploy and manage containers. Essentially, it abstracts installation of software and packages pertaining to your project and makes it
	easier to deploy, especially in a commercial setting.
	More information on Docker installation here: https://docs.docker.com/desktop/linux/install/#generic-installation-steps
	
	In order to run the project using docker, you will have to build the docker image using the "Dockerfile" provided:
		docker build -f Dockerfile -t [username/repository_name] .
	
	This will be similar to running the terminal option described above.
	
	Disclaimer: The execution of the Dockerfile was not tested as my PC does not have a GPU and Paperspace Gradient notebooks do not allow
	terminal access on free accounts. Owing to paucity of time, a workaround could not be figured as of writing this file. However, the
	past experience of the author enables him to claim that this option should work without much, if any, change to the Dockerfile.
	
	Disclaimer: The base image used in the Dockerfile is the same used by the corresponding notebook instance on Paperspace Gradient and
	available on Dockerhub: https://hub.docker.com/r/paperspace/nb-tensorflow/tags (last accessed: May 17 2022 21:54 IST)


------
EXTRAS
------
* Entire trained models (architecture + parameters) are available under "./saved_models/" - "naive_model/" has the model without FGSM
adversarial training, while "robust_model/" has the model trained using adversarial training technique. These checkpoints can be loaded
using instructions here: https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format

* Instance of executed code in the form of PDF file is available under "./extras/codebook/"

* Images of plots, original samples, adversarially-perturbed samples, etc. are available under "./extras/images/"
