---
datasets:
- microsoft/bleeding-edge-gameplay-sample
tags:
- wham
- microsoft
language:
- en
license_link: LICENSE.md
---
# World and Human Action Model (WHAM)
📄 [Paper](https://www.nature.com/articles/s41586-025-08600-3) • 🔗 [Sample Data](https://huggingface.co/datasets/microsoft/bleeding-edge-gameplay-sample)
<div align="center">
Anssi Kanervisto, Dave Bignell, Linda Yilin Wen, Martin Grayson, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Tabish Rashid, Tim Pearce, Yuhan Cao,
Abdelhak Lemkhenter, Chentian Jiang, Gavin Costello, Gunshi Gupta, Marko Tot, Shu Ishida, Tarun Gupta, Udit Arora,
Ryen W. White, Sam Devlin, Cecily Morrison, Katja Hofmann
</div><br>
<div align='center'>
Dynamic Generated Gameplay Sequence using WHAM. Showcasing diverse characters and actions across intricate maps.
  <div style="display: flex; flex-wrap: wrap;">
    <img style="width: calc(33.33%); margin-bottom: -35px;" src="assets/Readme/wham_gen_1.gif">
    <img style="width: calc(33.33%); margin-bottom: -35px;" src="assets/Readme/wham_gen_2.gif">
    <img style="width: calc(33.33%); margin-bottom: -35px;" src="assets/Readme/wham_gen_3.gif">
    <img style="width: calc(33.33%); margin-bottom: -35px;" src="assets/Readme/wham_gen_4.gif">
    <img style="width: calc(33.33%); margin-bottom: -35px;" src="assets/Readme/wham_gen_5.gif">
    <img style="width: calc(33.33%); margin-bottom: -35px;" src="assets/Readme/wham_gen_6.gif">
    <img style="width: calc(33.33%);" src="assets/Readme/wham_gen_7.gif">
    <img style="width: calc(33.33%);" src="assets/Readme/wham_gen_8.gif">
    <img style="width: calc(33.33%);" src="assets/Readme/wham_gen_9.gif">
  </div>
</div><br>
<div align='center'>
WHAM is capable of generating consistent, diverse, and persistent outputs, enabling various use cases for creative iteration.
<img style="width: 100%;" src="assets/Readme/model_capabilities.gif">
</div>

Muse is powered by a World and Human Action Model (WHAM), which is a generative model of gameplay (visuals and/or controller actions) trained on gameplay data of Ninja Theory’s Xbox game Bleeding Edge. Model development was informed by requirements of game creatives that we identified through a user study. Our goal is to explore the capabilities that generative AI models need to support human creative exploration. WHAM is developed by the [Game Intelligence group](https://www.microsoft.com/en-us/research/group/game-intelligence/) at [Microsoft Research](https://www.microsoft.com/en-us/research/), in collaboration with [TaiX](https://www.microsoft.com/en-us/research/project/taix/) and [Ninja Theory](https://ninjatheory.com/).

# Model Card

WHAM is an autoregressive model that has been trained to predict (tokenized) game visuals and controller actions given a prompt. Prompts here can be either visual (one or more initial game visuals) and / or controller actions. This allows the user to run the model in (a) world modelling mode (generate visuals given controller actions), (b) behavior policy (generate controller actions given past visuals), or (c) generate both visuals and behavior.

WHAM consists of two components, an encoder-decoder [VQ-GAN](https://compvis.github.io/taming-transformers/) trained to encode game visuals to a discrete representation, and a transformer backbone trained to perform next-token prediction. We train both components from scratch. The resulting model can generate consistent game sequences, and shows evidence of capturing the 3D structure of the game environment, the effects of controller actions, and the temporal structure of the game (up to the model’s context length).

WHAM was trained on  human gameplay data to predict game visuals and players’ controller actions. We worked with the game studio Ninja Theory and their game [Bleeding Edge](https://www.bleedingedge.com/) – a 3D, 4v4 multiplayer video game. From the resulting data we extracted one year’s worth of anonymized gameplay from 27,990 players, capturing a wide range of behaviors and interactions. A sample of this data is provided [here](https://huggingface.co/datasets/microsoft/bleeding-edge-gameplay-sample)

## Model Details

### Trained Models

In this release we provide the weights of two WHAM instances: 200M WHAM and 1.6B WHAM. Both have been trained from scratch on the same data set. 1.6B WHAM is evaluated in [our paper](https://www.nature.com/articles/s41586-025-08600-3). We additionally provide 200M WHAM as a more lightweight option for faster explorations.
-   [WHAM with 200M parameters](models/WHAM_200M.ckpt), model size: 3.7GB
-   [WHAM with 1.6B parameters](models/WHAM_1.6B_v1.ckpt), model size: 18.9GB

## Usage

### System Requirements

The steps below have been tested on the following setup:
- Linux workstation with Ubuntu 20.04.4 LTS
- Windows 11 workstation running WSL2 with Ubuntu 20.04.6 LTS

The current setup assumes that a CUDA-supported GPU is available for model inference. This has been tested on systems with `NVIDIA RTX A6000` and `NVIDIA A100` respectively. In addition, approximately `15GB` of free hard disk space is required for dowmloading the models.

The steps under Installation assume a python 3.9 installation that can be 
called using the command `python3.9` and the venv package for creating virtual environments. If either of these is not present, you can install this version of python under Ubuntu using:

```bash
sudo apt install python3.9
sudo apt install python3.9-venv
```

If you are using the WHAM Demonstrator, please ensure that you have the required [.NET Core Runtime](https://dotnet.microsoft.com/en-us/download/dotnet/7.0). If this is not yet installed, an error message will pop up from which you can follow a link to download and install this package.

### Installation

1. Clone this repository. We recommend starting without the large model files, using `GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:MSRTestOrg/WHAM`
2. `cd WHAM`
3. `./setup_local.sh`

This will set up a `python3.9` virtual environment and install the required packages (this includes packages required for the model server). The typical install time should be approximately 5 minutes.

4. Run `source venv/bin/activate` whenever you want to run model inference or the model server 

5. Download model from this HuggingFace repository (See note below): 
   1. Go to Files and versions and navigate to the `models` folder.
   2. Download the model checkpoint. The instructions below assume that the model checkpoints have been downloaded to your local `models` folder.

**Note:** On Linux systems, you can use `git clone` to clone the enire repository, including large files. Due to a limitation of `git lfs` on Windows, only files up to `4GB` are supported and we recommend downloading the model files manually from the `models` folder.


### Local Model Inference

This section assumes that you have followed the installation steps above.

(Optional) Download [sample data](https://huggingface.co/datasets/microsoft/bleeding-edge-gameplay-sample). For the local inference examples below, we recommend that you start with the `tiny-sample` set of only 4 trajectories for your initial exploration.

You can now run model inference to generate gameplay sequences as follows:

```python
python run_dreaming.py --model_path <path_to_checkpoint.ckpt> --data_path <path_to_sample_data_folder>
```

To run the 200M parameter (small) model (if you copied the tiny-sample folder to the root directory):

```bash
python run_dreaming.py --model_path models/WHAM_200M.ckpt --data_path tiny-sample
```

This uses the data in `data_path` as initial prompt sequences. The script will create a `dreaming_output` directory which will create two files per ground truth data file:
- An `.npz` file that contains a number of entries, most important of which are:
  - `encoded_decoded_ground_truth_images`: the original context images, encoded and decoded with the VQGAN.
  - `dreamt_images`: the sequence of all dreamt images.
- An `.mp4` file of the context data + dreamt images for easier viewing.

This requires approximately 4.5GB of VRAM on a single A6000, but only uses batch size of one. To speed up the process, increase batch size with `--batch_size` argument. With a single A6000 and `--batch_size 12` this uses approximately 30GB of VRAM. Generating gameplay sequences from the full 512 video dataset takes around 24 hours.

Please note that the first output from the script is generated when the first gameplay sequence has been generated. This may take several minutes when using an `A6000` GPU, or longer for older generation GPUs.

See python `run_dreaming.py --help` for different settings.

### WHAM Demonstrator

#### Setting up the Model Server

We have tested the server code as provided on a single Linux machine with four `A6000 GPUs` (large model) as well as on a Windows machine running Ubuntu under `WSL2`, equipped with a single `GeForce GTX 1080` (small model). Model inferences can be run on lower spec NVIDIA GPUs by reducing the batchsize.

The steps below assume that the installation steps above have been followed and that the model files have been downloaded to your local machine.

In your terminal, activate the newly installed virtual environment (if it isn't already):

```bash
source venv/bin/activate
```

Start the server, pointing it to the model:

```bash
python run_server.py --model <path_to_model_file>
```

To run the 200M parameter (small) model:

```bash
python run_server.py --model models/WHAM_200M.ckpt
```

To run the 1.6B parameter (large) model:

```bash
python run_server.py --model models/WHAM_1.6B_v1.ckpt
```


The server will start and by default listen on localhost port 5000 (this can be configured with `--port <port>`).

**Note:** If you run out of VRAM when running the server, you can reduce the `MAX_BATCH_SIZE` variable in `run_server.py`.


#### Install the WHAM Demonstrator App (Windows only)

After cloning or downloading this repository, navigate to the folder `wham/wham_demonstrator`, and start the Windows application `WHAMDemonstrator.exe` within that folder.

Follow the instructions in the provided README.md within WHAM Demonstrator to connect to your model server and get an overview of supported functionality.


## Intended Uses

This model and accompanying code are intended for academic research purposes only. WHAM has been trained on gameplay data from a single game, Bleeding Edge, and is intended to be used to generate plausible gameplay sequences resembling this game.

The model is not intended to be used to generate imagery outside of the game Bleeding Edge. Generated images include watermark and provenance metadata. Do not remove the watermark or provenance metadata..

WHAM can be used in multiple scenarios. The following list illustrates the types of tasks that WHAM can be used for:
- World Model: Visuals are predicted, given a real starting state and action sequence.
- Behaviour Policy: Given visuals, the model predicts the next controller action.
- Full Generation: The model generates both the visuals and the controller actions a human player might take in the game.

## Training

### Model

- Architecture: A decoder-only transformer that predicts the next token corresponding to an interleaved sequence of observations and actions. The image tokenizer is a VQ-GAN.
- Context length: 10 (observation, action) pairs / 5560 tokens
- Dataset size: The model was trained on data from approximately `500,000` Bleeding Edge games from all seven game maps (over 1 billion observation, action pairs 10Hz, equivalent to over 7 years of continuous human gameplay). A data sample is provided in [bleeding-edge-gameplay-sample](https://huggingface.co/datasets/microsoft/bleeding-edge-gameplay-sample). This is the test data used for our evaluation results, and has the same format as the training data.
- GPUs: 98xH100 GPUs
- Training time: 5 days

### Software

- [PyTorch Lightning](https://github.com/pytorch/pytorch)
- [Flash-Attention](https://github.com/HazyResearch/flash-attention)
- [ffmpeg](https://github.com/FFmpeg/FFmpeg)
- [exiftool](https://github.com/exiftool/exiftool)

## Bias, Risks and Limitations

- The training data represents gameplay recordings from a variety of skilled and unskilled gameplayers, representing diverse demographic characteristics. Not all possible player characteristics are represented and model performance may therefore vary.
- The model, as it is, can only be used to generate visuals and controller inputs. Users should not manipulate images and attempt to generate offensive scenes.   

### Technical limitations, operational factors, and ranges

Model: 
-	Trained on a single game, very specialized, not intended for image prompts that are out of context or from other domains
-	Limited context length (10s)
-	Limited image resolution (300px x 180px), the model can only generate images at this fixed resolution.
-	Generated images and controls can incorrect or unrecognizable.
-	Inference time is currently too slow for real-time use.

WHAM Demonstrator:
-	Developed as a way to explore potential interactions. This is not intended as a fully-fledged user experience or demo.

Models trained using game data may potentially behave in ways that are unfair, unreliable, or offensive, in turn causing harms. We emphasize that these types of harms are not mutually exclusive. A single model can exhibit more than one type of harm, potentially relating to multiple different groups of people. For example, the output of the model can be nonsensical or might look reasonable but is inaccurate with respect to external validation sources.
Although users can input any image as a starting point, the model is only trained to generate images and controller actions based on the structure of the Bleeding Edge game environment that it has learned from the training data.    Out of domain inputs lead to unpredictable results. For example, this could include a sequence of images that dissolve into unrecognizable blobs   .  
Model generations when “out of scope” image elements are introduced will either:
- Dissolve into unrecognizable blobs of color.
- Morphed into game-relevant items such as game characters. 

## Evaluating WHAM
WHAM is evaluated based on its consistency, diversity, and persistency. Consistency is measured using Fréchet Video Distance (FVD), while diversity is assessed by comparing the marginal distribution of real human actions to those generated by the model using the Wasserstein distance. Persistency is tested using two scenarios: by adding a static power-up object to a game visual and by adding another player character to a game visual used for prompting the model. For detailed evaluation results, see the paper that [introduces the model](https://www.nature.com/articles/s41586-025-08600-3). 

### Responsible AI testing
WHAM has been tested with out of context prompt images to evaluate the risk of outputting harmful or nonsensical images. The generated image sequences did not retain the initial image, but rather dissolved into either unrecognizable blobs or to scenes resembling the training environment. 


## License

The model is licensed under the [Microsoft Research License](LICENSE.md)

this work has been funded by Microsoft Research

## Privacy & Ethics Statement

[Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839)

## Trademark Notice

**Trademarks** This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## Contact Information  
For questions please email to muse@microsoft.com