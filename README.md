# convolutional-neural-markov-model
Experiment codes for the paper '2D Convolutional Neural Markov Models for Spatiotemporal Sequence Forecasting'

# Project structure
## Main
|-- configs                  : Configurations for data generation and training model  
|-- data_generation         : Data generation script  
|-- models                  : Training model codes - ConvLSTM, ConvDMM, Vanilla DMM  
|-- utils                   : Utils codes  
scripts                     : Training scripts  
generate_data.py            : Data generation  
visualize                   : 2D and 3D visualization script  

## Generated by scripts
|-- data                    : Synthetic data generated by the data generation script

# Generate synthetic data
Use
```
python generate_data.py
```
to generate heat diffusion data, which parameters are defined by a sample configuration file in ```/configs/data_generation```. Description for each parameters are as follows. The types of probability distribution function available are ```normal``` and ```cauchy```.

| Parameter                  | Description                                                                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| data_path                  | Path of the generated data. Remember that the root folder of the saved data will be defined by the ```run``` parameter defined in ```hydra.run```. Change this part to generate both training and validation dataset. |
| n_simulations              | Number of simulations.                                                                                                                                                                                                |
| simulation_length          | Length of the simulation (timestep).                                                                                                                                                                                  |
| n_samples                  | Number of samples taken from each simulation. This means the total number of data generated by the script will be ```n_simulations * n_samples```                                                                     |
| sample_length              | Length of the sample.                                                                                                                                                                                                 |
| sample_time_difference     | Time difference of the sample. When sampling a snippet data from a simulation, we allow time-skipping which timeskip difference is defined by this parameter.                                                         |
| temp_base                  | Base temperature (temperature not in the generated circle of heat).                                                                                                                                                   |
| temp_low                   | Minimum temperature range of the circle of heat.                                                                                                                                                                      |
| temp_high                  | Maximum temperature range of the circle of heat.                                                                                                                                                                      |
| radius_min                 | Minimum radius range of the circle of heat.                                                                                                                                                                           |
| radius_max                 | Maximum radius range of the circle of heat.                                                                                                                                                                           |
| center_x_min, center_y_min | Minimum central x, y position range of the circle of heat.                                                                                                                                                            |
| center_x_max, center_y_max | Maximum central x, y position range of the circle of heat.                                                                                                                                                            |
| transition_loc             | Mean of the transition noise.                                                                                                                                                                                         |
| transition_scale           | Variance of the transition noise.                                                                                                                                                                                     |
| transition_noise_type      | Type of probability distribution function governing the transition noise.                                                                                                                                             |
| emission_loc               | Mean of the emission noise.                                                                                                                                                                                           |
| emission_scale             | Variance of the emission noise.                                                                                                                                                                                       |
| emission_noise_type        | Type of probability distribution function governing the emission noise.                                                                                                                                               |

# Running
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