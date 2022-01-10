# climate-pattern-modeling-with-hmm  

This repository contains all the programs coded for the assignment on climate pattern modeling with Hidden Markov Model (HMM) **(Offline-2)**. In this assignment, the climate pattern of a tropical Eastern Pacific Ocean region is modeled using HMM. The HMM parameters are estimated with rainfall estimate observations and the most likely climates sequence is estimated separately with both provided and learned HMM parameters. Viterbi algorithm is implemented for solving decoding problem and Baum-Welch learning algorithm is implemented for solving HMM parameters estimation problem.  



## navigation  

- `inputdir/` folder contains two input files `data.txt` and `parameters.txt` providing observations and HMM parameters respectively.  

- `spec/` folder contains tasks specification for this particular assignment.  
- `src/` folder contains a Jupyter notebook (with `.ipynb` extension) and a Python script (with `.py` extension obviously) both containing implementation of Viterbi as well as Baum-Welch learning algorithms and code for inputs preprocessing.  



## input files description  

- `data.txt` file contains rainfall estimate observations for the last 1000 years.  
- `parameters.txt` file contains predetermined HMM parameters.  



## getting started  

In order to run the Python script, place `hmm_climate_pattern_modeling.py` file inside a workspace folder. Create or place `inputdir/` folder inside the same workspace folder and place `data.txt` and `parameters.txt` files inside `inputdir/`. You may need to install some Python modules beforehand. Run `hmm_climate_pattern_modeling.py` inside the workspace folder for running the main program.  



## output files description  

After running the main program, `outputdir/` folder will be created inside the aforementioned workspace folder. This folder will contain the following output files.  

- `estimated-states-sequence-with-provided-parameters.txt` file contains the estimated states (either **El Nino** or **La Nina**) sequence determined for the predetermined HMM parameters from `parameters.txt`.  
- `learned-parameters.txt` file contains the estimated HMM parameters determined using Baum-Welch learning algorithm.  
- `estimated-states-sequence-with-learned-parameters.txt` file contains the estimated states (either **El Nino** or **La Nina**) sequence determined for the learned HMM parameters outputted in `learned-parameters.txt`.  

