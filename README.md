# Ant Simulation codes

## Lattice Ant Model

This simulation is based on [Non-stationary aging dynamics in ant societies](https://www.sciencedirect.com/science/article/pii/S0022519311002347). The only thing that is changed is that the model in the paper uses toroidal surface whereas this code is a bounded rectangle with boundary effects.

Requires parameter file `lattice_ant_params.json` to supply parameters:

* `boundary_condition` : Whether or not to enable boundary condition
* `ant_selection_method` : Method for cycling through the ants. It can be random or sequential.
* `enable_wall_effect` : Enables ant-boundary interaction.
* `wall_interaction_strength` : Wall interaction strength
* `timesteps` : Total timesteps for simulation
* `montage_step` : How many timesteps to skip between montage rendering
* `lattice_size` : Lattice size
* `n_species` : Number of species (different types of workers)
* `m_count` : Number of ants for each species given as an `1 x n` array
* `interaction_table` : `n x n` array of interaction strengths between different species
* `length_scale` : Size of the bin in units of body length of ant
* `threshold` : Threshold value for random fluctuation to try to exceed
* `energy_scale` : Energy scale factor used for metropolis
* `temperature` : "Temperature" value used for metropolis

## Continuous-Time Correlated Random Walk

Work in progress
