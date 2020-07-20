# convolutional-neural-markov-model
Experiment codes for the paper '2D Convolutional Neural Markov Models for Spatiotemporal Sequence Forecasting'

# Project structure
|-- data_generation         : Data generation script  
|-- models                  : Training model codes - ConvLSTM, ConvDMM, Vanilla DMM  
|-- utils                   : Utils codes  
scripts                     : Training scripts  
generate_data.py            : Data generation  
visualize                   : 2D and 3D visualization script  

# Running
Use
```
python generate_data.py
```
to generate heat diffusion data, which will be saved at the ```/data/diffusion/train``` or ```/data/diffusion/valid```, depending on the context.

Every training scripts start with ```train_```, followed by the name of the model and the data to be experimented upon. For example,
```
python train_cnmm_heatmap.py -tn 0 -ee 150
``` 
will start CNMM training with specified hard-coded configurations and data location inside the script, with heat diffusion data as target. The ```-tn``` and ```-ee``` means trial number and last training epoch number, respectively. The detailed arguments for each scripts can be displayed by running:
```
python train_cnmm_heatmap.py --help
```

In the near future, we will upload the codes and guidelines for testing with custom data.

# TODOs
As can be seen, the training scripts are divided into several different scripts for each settings. This is very ineffective, as a change to the flow of the script will require us modifying other affected scripts. We plan to rewrite the training scripts to increase efficiency and effectiveness. We also plan to refactor/reorganize the whole project into a much more manageable structure.

# Acknowledgements
We referenced and utilized [Pyro's DMM code](https://pyro.ai/examples/dmm.html) as a baseline code to model our approach.

# Author
Calvin Janitra Halim