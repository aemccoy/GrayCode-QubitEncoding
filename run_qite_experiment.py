############################################################################################################
# Obtain energies for deuteron using quantum imaginary time evolution.
# This program takes a required yaml input file, e.g. parameters_qite.yml, and outputs
#   <basename>.npy : File containing deuteron ground state energy for delta_time step
#   <basename> generated based on input parameters.
#       qite_results_Nmax{Nmax}_{interaction}_{encoding}_{backend}.npy
#
# Parameters in input file:
#
#   Mandatory parameters:
#      Nmax (int) -- Nmax of HO basis
#
#     Optional parameters:
#       interaction (string) -- tag for which interaction to use for deuteron Hamiltonian.
#           Supported interactions are:
#               "toy" (default) : Toy Hamiltonian used in Oakridge paper 
#               "N4LOsrg1.5" : N4LO interaction for hw=14, srg evolved with lambda=1.5
#               "Daejeon" : Daejeon16 interaction for hw=15
#               "N3L0srg2.15" : N3LO interaction for hw=20, srg evolved with lambda=2.15
#               "N2LOopt" : N2LOopt interaction with hw=20
# 
#       encoding (string) -- specifies encoding of deuteron Hamiltonian.  See paper [XREF]
#          Supported encodings are:
#             'gray_code'  (default)
#             'jordan_wigner'
# 
#       initialization(string)--starting state of circuit.  Supported initalizations are 
#           'single_State' (default) : Initialized to |000...> for Gray code and |0...01> for Jordan Wigner
#           'uniform' : uniform superposition of all states.  Only valid for Gray code
#           <np.array> : Normalized numpy array representing starting vector 
# 
#       N_trials (int) -- number of independent trials to run.  (Default value : 1).
#
#       N_time_steps (int) -- number of time steps to take in evolution. (Default value : 1)
# 
#       delta_time (float) -- time of evolution for every time step (Default value : 0.01)      
# 
#       merge_step (int,None) -- Number of steps after which to merge sequence of evolution operators 
#                                   into a single operator (Default value : None)
#           
#       threshold (float) -- threshold below which eigenvalues of A matrix considered to be zeros
#               cond arg in scipy lstsq solver
# 
#       backend (string) -- name of qiskit backend.  Supported backend are
#          'statevector_simulator'     (default)
#          'qasm_simulator'
#
#       N_shots (int) -- number of repetitions of each circuit.  Default value : 10000).
#
#       device_name (string) -- Handle for noise model used in qasm simulations based on IBMQ machines.
#          If handle is 'None' or no handle is given then no noise model is used.  Files containing data
#          used to create noise models are found in the directory "devices".
#          Valid handles are:
#              'None'              (default)
#              'ibmq_16_melbourne'
#              'ibmq_5_yorktown'
#              'ibmq_burlington'
#              'ibmq_essex'
#              'ibmq_london'
#              'ibmq_vigo'
#
#      mitigate_meas_error (boolean) : Apply readout error mitigation (Default value : False).
#
#      N_cpus (int) -- Number of processes over which trials are distributed. (Default value : 1)
#
#      output_dir (string) -- Directory for results files.  (Default value : "outputs").
#                             Directory created at runtime if it doesn't exist.
#
#       layout (?)
############################################################################################################

import sys
sys.path.append("src/")
import os
import yaml

import numpy as np
np.warnings.filterwarnings('ignore')
np.set_printoptions(precision=6, suppress=True)

from pprint import pprint
from tqdm import tqdm
# import itertools


from qiskit.tools import parallel_map

from hamiltonian import *
from utils import *
from qiskit_circuits import *
from qite import *




if __name__ == "__main__":

    # Set to true if text output desired in addition to numpy outputfiles
    save_to_textfile=True

    # Get the input parameters filename
    if len(sys.argv) != 2:
        raise ValueError("Syntax: run_qiskit_experiment.py <input parameters>\n")

    parameter_file = sys.argv[1]
    parameters = {}

    # Read the parameters
    with open(parameter_file) as infile:
        try:
            parameters = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)

    # If parameters not read in from file, set parameters to default.
    parameters=set_qite_parameters(parameters)

    ###############################################################################################
    # Create array of dummy variable for each trial (required by parallel_map)
    trial_indices=[i for i in range(parameters['N_trials'])]
    # Zero initialize energy array
    results=[None]*parameters['N_trials']
   
    # Run experiment or experiment simulation for N_trials to compute eigenvalues of Hamiltonian <ham>.  
    # Runs are distributed over <N_cpus> processes using qiskit's built-in parallel map function.
    results = parallel_map(
        do_qite_experiment, trial_indices, task_args=[parameters], num_processes=parameters['N_cpus']
        )

    encoding_label="".join(parameters["encoding"].split("_"))
    backend_label=parameters["backend"].split("_")[0]
    Nmax=parameters["Nmax"]
    interaction=parameters["interaction"]
    merge_step=parameters["merge_step"]
    output_filename=f"qite_results_Nmax{Nmax:02d}_{interaction}_{encoding_label}_{backend_label}_ms{merge_step}"
    print(output_filename)
    np.save(output_filename,results)
    results_in=np.load(output_filename+".npy")
    # print(results)
    # print(results_in)
 