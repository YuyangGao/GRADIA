# GRADIA - Aligning Eyes between Humans and Deep Neural Network through Interactive Attention Alignment

The Pytorch implementation of GRADIA model for CSCW'22 paper: [Aligning Eyes between Humans and Deep Neural Network through Interactive Attention Alignment](https://arxiv.org/pdf/2202.02838.pdf)

## Desciption
This codebase proivdes the necessary running environment (including the human explanation label) to train and evaluate the proposed GRADIA model on the Gender classification image datasets. 

## Running Environment

Python pakage requirement:
- python==3.7.9
- pytorch==1.5.0
- torchvision==0.6.0
- opencv==4.5.0
- numpy==1.16.5

## Data Preparation & Description

1. Download the datasets as well as our human explanation labels via google drive link below:

* Gender classification dataset: [https://drive.google.com/file/d/1CSPzEUcd8aCiiQCspO5oKzqHmS8tY_85/view?usp=sharing](https://drive.google.com/file/d/1CSPzEUcd8aCiiQCspO5oKzqHmS8tY_85/view?usp=sharing)
* Our human explanation labels: [https://drive.google.com/file/d/1tJoahurhyhNNlmBdwJlMi3-vxXz2nxkY/view?usp=sharing](https://drive.google.com/file/d/1tJoahurhyhNNlmBdwJlMi3-vxXz2nxkY/view?usp=sharing)

2. Extract the dataset and place them in 'gender_data/' data directory.

3. Extract the explanation labels and place them in the root directory of the project.

The data in the folders are mostly self-explained by their names, but just to provide a bit more info here:
*  **train**: this folder contains our training set
*  **val**: this folder contains our validation set
*  **test** : this folder contains our test set
*  **p2_train** : this folder contains our phase 2 fine-tuning set, which is the combination of orignal training set and those selected sample from validation set based on reasonablity matrix. Notice that you may need to build up your own phase 2 training sample set, as it is subject to be dependent on the model trained in phase 1 as well as the specific strategy for selecting 'what to adjust'. We provide this folder here mainly for the reproducablity of the expeirmental results shown in our paper.

For more information about the dataset or experiment setup, please refer to the experimental section in the paper.

## Sample Training Scripts 

* Phase 1: train the model on training set, the trained model will be saved by default as 'model_out' in /model_save and will be used later in phase 2 finetuning

```
python GRADIA.py --train-batch 100 --test-batch 10 --n_epoch 50
```

* Phase 2: load & finetune the phase 1's model on val+train set with all available attention labels

```
python GRADIA.py --train-batch 100 --test-batch 10 --n_epoch 50 --model_name model_out --trainWithMap
```

*Notice*: The above code will introduce the attention loss on all the samples that have attention label available. This comes in handy if you don't want any additional human assesssment on reasonbality. This only require you to have the attention labels for val&test set.

However, if you do get the reasonablity matrix on validation set of your phase 1's model (such as the 'reasonablity_val.json' provided), you can selectively apply the attention loss only on those 'unreasonable' or 'inaccurate' samples based on the matrix (this is the model version shown in our paper)

To do so, simply add '--reasonablity' into the command:

```
python GRADIA.py --train-batch 100 --test-batch 10 --n_epoch 50 --model_name model_out --trainWithMap --reasonablity
```

## Sample Testing Scripts 
To test the performance of the specified model in '--model_name [your_model_name]', just add '--evaluate' into the command:

```
python GRADIA.py --train-batch 100 --test-batch 10 --model_name [your_model_name] --evaluate
```


## Result Validation

Where to look for the results:
* The overall model performance can be find directly in the program output in Console
* Explanation visualization can be find in 'attention/' folder

Below are some sample explanations visualization results on GRADIA and other comparison methods. The model-generated explanations are represented by the black-white mask overlaid on the original image samples, where more importance is given to more transparent area.

<img src="https://github.com/YuyangGao/GRADIA/blob/main/example_figs/S2_results2.png" alt="drawing" width="1500"/>

If any further questions, please feel free to reach out to us via email yuyang.gao@emory.edu

##

And if you find this repo useful in your research, please consider cite our paper:

    @article{gao2022aligning,
    title={Aligning Eyes between Humans and Deep Neural Network through Interactive Attention Alignment},
    author={Gao, Yuyang and Sun, Tong Steven and Zhao, Liang and Hong, Sungsoo Ray},
    journal={arXiv preprint arXiv:2202.02838},
    year={2022}
    }
