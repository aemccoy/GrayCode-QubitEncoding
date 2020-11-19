import numpy as np 
from scipy.linalg import lstsq
from pprint import pprint
from tqdm import tqdm

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute, Aer
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.initial_states import Custom

from device import *
from hamiltonian import *
from utils import *
from qiskit_circuits import *
from qlanczos import *
import itertools

def set_qite_parameters(parameters, verbose=True):
    """
    Set any parameters not included in input parameter file to default values.

    Parameters:
        parameters (dictionary) : Input parameters for VQE.

    Returns:
        parameters : Updated parameters to include default parameters not given by input file
    """
    default_parameters = {
                      'Nmax' : 0,
                      'interaction':"toy",
                      'encoding' : 'gray_code',
                      'initialization' : 'single_state',
                      'N_trials' : 1,
                      'N_time_steps': 1,
                      'delta_time' : .01,
                      'merge_step' : None,
                      'qlanczos' : False,
                      'krylov_threshold' : 0.99, ## thresold for QLanczos regularization factor
                      'qite_threshold' : 1e-2, ## threshold for lstsq solver
                      'qlanczos_threshold':1e-2, ## threshold for eigenvalues set to zero
                      'backend' : 'statevector_simulator',
                      'N_shots' : 10000,
                      'device_name' : None,
                      'mitigate_meas_error' : False,
                      'layout' : None,
                      'show_progress' : True,
                      'number_cnot_pairs': 0,
                      'number_circuit_folds': 0,
                      'zero_noise_extrapolation': False,
                      'degree_extrapolation_polynomial':1,
                      'N_cpus' : 1,
                      'output_dir': 'outputs'
                      }

    # Check for valid encoding and backend
    for param in parameters:
        if (param == 'encoding') and (parameters['encoding']!='gray_code' and parameters['encoding']!='jordan_wigner'):
            raise ValueError("Encoding {} not supported.  Please select 'gray_code' or 'jordan_wigner'. ".format(parameters['encoding']) )
        
        if (param == 'backend') and (parameters['backend']!='statevector_simulator' and parameters['backend']!='qasm_simulator'):
            raise ValueError("Backend {} not supported.  Please select 'statevector_simulator' or 'qasm_simulator'. ")

    # Set default values for anything not provided
    for parameter in default_parameters.keys():
        # parameter 'N_states' must be set in input file
        if 'Nmax' not in parameters.keys():
            raise ValueError("Must provide Nmax for simulation.")
        if parameter not in parameters.keys():
            # Setting parameter to default parameter
            parameters[parameter] = default_parameters[parameter]
            if verbose: 
                print(f"No value for parameter {parameter} provided.")
                print(f"Setting {parameter} to default value {default_parameters[parameter]}.")

    # accept 'None' as a string in params.yml
    if parameters['device_name'] == 'none' or parameters['device_name'] == 'None':
        parameters['device_name'] = None
    if parameters['layout'] == 'none' or parameters['layout'] == 'None':
        parameters['layout'] = None

    # Check compatibility between device, layout and measurement error mitigation
    if parameters['device_name'] is None:
        if parameters['layout'] is not None:
            raise ValueError("Layout cannot be specified without a device.")
        if parameters['mitigate_meas_error'] is not False:
            raise ValueError("Measurement mitigation is not possible if no device is specified")

    # Layout must None or a list of ints
    if parameters['layout'] is not None:
        assert type(parameters['layout']) is list, "Layout must be a list of integers."
        assert all([type(q) is int for q in parameters['layout']]), "Layout must be a list of integers."

    if verbose:
        print("\nExperiment parameters")
        pprint(parameters)
        print()

    return parameters



def sigma_terms(n_qubits,encoding="gray_code"):
    """
    Get all possible pauli operators with an odd number of Y gates for n_qubit qubits 
    
    Input:
        n_qubits (int) : number of qubits 

    Returns:
        sigmas (list<WeightedPauliOperator>) : List of pauli operators with an odd number of Y

    """ 
    pauli_strings=["I","X","Y","Z"]
    pauli_product_strings=list(map(lambda x : "".join(x),list(itertools.product(*[pauli_strings for i in range(n_qubits)]))))

    ## Extract out pauli terms of odd number of Y 
    paulis=[]
    for string in pauli_product_strings:
        if string.count("Y")%2==1:
            ## If JW encoding, include only operators which enable evolution operator to stay in "one-hot" subspace 
            ## These are operators with an even number of Y and X gates 
            if encoding == "jordan_wigner":
                if (string.count("Y")+string.count("X"))%2 != 0:
                    continue
            ## Add Pauli operator to list 
            paulis.append(Pauli.from_label(string))

    sigmas=[]
    for pauli in paulis:
        sigma=[(1.0,pauli)]
        sigmas.append(WeightedPauliOperator(sigma))

    return sigmas

    
    
def b_terms(H,sigmas):
    """
    Compute b=i[H,sigma]

    Input:
        H (hamiltonian.EncodingHamiltonian) : Qubit Hamiltonian
        sigmas (list<WeightedPauliOperator>) : List of pauli operators with an odd number of Y
            
    Returns:

        b (dictionary) : b operator expressed as dictionary of {label:coef}, each label is a pauli string.
    """
    b_pauli_terms=[]
    # H_paulis=H_weighted_paulis(H)
    H_paulis=H.weighted_pauli
    for sigma in sigmas:
        product=1j*(H_paulis.__mul__(sigma)-sigma.__mul__(H_paulis))
        product.chop(1e-5)
        terms=get_pauli_terms(product)
        b_pauli_terms.append(terms)
    return b_pauli_terms

## define S array
def S_terms(sigmas):
    """
    Compute operators S_{ij}=2sigma_i.sigma_j 

    Input:   
        sigmas (list<WeightedPauliOperator>) : List of pauli operators with an odd number of Y

    Returns:
        S (dictionary) : S operator expressed as a dictionary of dictionaries.  
                Outer dictionary keyed by (i,j)
                Inner dicitonary {label:coef}: S_{ij} expanded in terms of pauli operators 
 
    """
    d=len(sigmas)
    S={}
    for i in range(d):
        sigma1=sigmas[i]
        for j in range(d):
            sigma2=sigmas[j]
            product=sigma1.__mul__(sigma2)
            S[(i,j)]=get_pauli_terms(product)
    return S



def get_intersection_pauli_terms(H,b_pauli_terms,S_pauli_terms):
    """
    Get intersection of pauli terms needed to compute S and b
    
    Only useful if not computing all possible gate combinations
    """
    ## Initialize set with pauli terms in Hamiltonian
    pauli_set=set(H.pauli_coeffs.keys())
    
    ## Add b 
    for bI in b_pauli_terms:
        pauli_set.update(bI.keys())

    ## Add S
    for term in S_pauli_terms.values():
            pauli_set.update(term.keys())

    return pauli_set

def A_pauli_operator(
        delta_time,sigmas,S_pauli_terms,b_pauli_terms,
        expectation_values,Ccoef,A_threshold,verbose=False
    ):
    """
    Hermitian operator such that exp[-iAt]|psi>~exp[-Ht]|psi>
    """
    
    n_qubits=sigmas[0].num_qubits

    ## compute b
    num_sigmas=len(sigmas)
    b=np.zeros(num_sigmas)
    for sigma_idx in range(num_sigmas):
        bI=b_pauli_terms[sigma_idx]
        for term in bI:
            b[sigma_idx]+=bI[term]*expectation_values[term]

    b=b/np.sqrt(Ccoef)
          
    ## Comput S
    Smatrix=np.asmatrix(np.zeros((num_sigmas,num_sigmas)))
    for i in range(num_sigmas):
        for j in range(num_sigmas):
            Sij=S_pauli_terms[(i,j)]
            for term in Sij:
                #(S+S^T) gives rise to factor of 2
                Smatrix[i,j]+=2*Sij[term]*expectation_values[term] 

    ## Solve for a in (S+S^T)a=b
    a,_,_,_=lstsq(Smatrix,b,cond=A_threshold)
    if verbose==True:
        print("Smatrix")
        print(Smatrix)
        print("b\n",b)
        print("a\n",a)
    
    ## Construct A from a
    identity_string="I"*n_qubits

    ##Null initialize A with 0.0*Identity. 
    A=WeightedPauliOperator([(0.0,Pauli.from_label(identity_string))])

    for i in range((len(sigmas))):
        # A+=delta_time*a[i]*sigmas[i]
        A+=a[i]*sigmas[i]

    if verbose==True:       
        print("t=",t)
        print("-----------------------------------------")

        print("wavefunction\n",wavefunction)        
        print("\nA operator")
        print(A.print_details())
        
        A_pauli_terms=get_pauli_terms(A)
        Amatrix=reduce(lambda x,y: x+y,[A_pauli_terms[term]*get_pauli_matrix(term) for term in A_pauli_terms])
        print(Amatrix)
        print("-----------------------------------------")
    
    return A


def run_circuit_statevector(
        n_qubits,
        evolution_operators,
        time,
        initialization=None,
        encoding="gray_code",
        number_cnot_pairs=0,
        number_circuit_folds=0
    ):
        ## If num_cnot_pairs>0, then do extrapolation is cnot, 
        ## otherwise if number_circuit_folds>0 do extrapolation by folding the circuit
        if number_cnot_pairs!=0:
            number_circuit_folds=0
        if number_circuit_folds!=0:
            number_cnot_pairs=0        

        ## Initalize circuit
        q = QuantumRegister(n_qubits)
        c = ClassicalRegister(n_qubits)
        circuit=initialize_circuit(q,c,initial_state=initialization,encoding=encoding)

        ## If evolution_operators given, evolve cirucit exp(-iAt) for each A in evolution operators
        if len(evolution_operators)>0:
            circuit=append_evolution_circuit(q,evolution_operators,time,circuit,number_cnot_pairs)

        ## Apply folding if number_circuit_folds>0
        circuit=fold_circuit(circuit,number_circuit_folds=number_circuit_folds)

        ## Execute circuit
        job = execute(circuit, backend=Aer.get_backend("statevector_simulator"),shots=1)
        
        return job.result().get_statevector(circuit)        

    
def run_circuit_qasm(
        n_qubits,
        evolution_operators,
        pauli_id,
        time,
        n_shots=100000,
        initialization=None,
        encoding="gray_code",
        noise_model=None,
        device=None,
        number_cnot_pairs=0,
        number_circuit_folds=0
        # extrapolation=None,
        # num_extrapolation_steps=0
    ):

        # # if extrapolation == None:
        # #     assert(num_extrapolation_steps==0)
        
        # # noisy_counts={}
        # # xs=[]
        # # number_cnot_pairs=0
        # # number_circuit_folds=0
        # # for n in range(num_extrapolation_steps+1):

        #     if extrapolation == "cnot_pairs": number_cnot_pairs=n
        #     elif extrapolation == "circuit_folding": number_circuit_folds=n

        ## Initalize circuit
        q = QuantumRegister(n_qubits)
        c = ClassicalRegister(n_qubits)
        
        circuit=initialize_circuit(q,c,initial_state=initialization,encoding=encoding)

        ## If evolution_operators given, evolve cirucit exp(-iAt) for each A in evolution operators
        if len(evolution_operators)>0:
            circuit=append_evolution_circuit(q,evolution_operators,time,circuit,number_cnot_pairs)
        
        ## Rotate to measurement basis 
        circuit=append_measurement_basis_rotation(circuit,q,pauli_id) 

        ## Fold circuit if number_circuit_folds>0 
        circuit=fold_circuit(circuit,number_circuit_folds=number_circuit_folds)   

        ## Add final measurement
        circuit.measure(q,c)

        ## Execute circuit
        if noise_model != None:
            job = execute(
                    circuit, 
                    backend=Aer.get_backend("qasm_simulator"),
                    shots=n_shots,
                    noise_model=noise_model,
                    basis_gates=noise_model.basis_gates,
                    optimization_level=0
                    )
        elif device != None:
            job = execute(
                    circuit, 
                    backend=Aer.get_backend("qasm_simulator"),
                    shots=n_shots,
                    noise_model=device.noise_model,
                    basis_gates=device.noise_model.basis_gates,
                    coupling_map=device.coupling_map,
                    initial_layout=device.layout,
                    optimization_level=0
                    )
        else:
            job = execute(circuit, 
                backend=Aer.get_backend("qasm_simulator"),
                shots=n_shots,
                optimization_level=0
                )

        
        ## Get counts
        counts=job.result().get_counts(circuit)
        ## populate dictionary for extrapolation 
        
                
        #     for state in counts:
        #         if state not in noisy_counts:
        #             noisy_counts[state]=np.zeros(num_extrapolation_steps+1)
                
        #         noisy_counts[state][n]=counts[state]
                
        #     xs.append(2*n+1)

        # ## if extrapolation not None, do extrapolation for each set of counts 
        # if extrapolation != None:
        #     for state in noisy_counts:
        #         ## normalize counts 
        #         noisy_counts[state]=[count/n_shots for count in noisy_counts[state]]
        #         print(noisy_counts[state])
        #         coef=np.polyfit(xs,noisy_counts[state],1)
        #         linear_fit=np.poly1d(coef)
        #         print("extrapolated value ",linear_fit(0))
        #         ## Evaluate linear_fit polynomial at zero to get extrapolated number of counts
        #         ## Replace value in counts with extrapolated value 
        #         counts[state]=linear_fit(0) 
        
        # ## if extraplation is None, then counts is simply the counts dictionary evaluated 
        # ## in the circuit, which is only evaluated one time

        # else:
        ##normalize counts
        for state in counts:
            counts[state]=counts[state]/n_shots
    
        return counts

###########################################################################################
def qite_experiment(H,parameters,verbose=False):
    """
    Run qite evolution to get energies of ground state 
    """
    ### Extract values from parameters
    time_steps = parameters['N_time_steps']
    delta_time = parameters['delta_time']
    backend = parameters['backend']
    initialization = parameters['initialization']
    encoding = parameters['encoding']
    A_threshold = parameters['qite_threshold']
    merge_step = parameters['merge_step']
    show_progress=parameters['show_progress']
    n_shots=parameters['N_shots']
    QLanczos=parameters['qlanczos']
    do_extrapolation=parameters['zero_noise_extrapolation']
    number_cnot_pairs=parameters['number_cnot_pairs']
    number_circuit_folds=parameters['number_circuit_folds']
    
    ## If do_extrapolation == True, identify extrapolation type
    ## TODO fix so that if do extrapolation is false, num_cnot_pairs etc.
    ## still carried through to the end. 
    extrapolation=None
    num_extrapolation_steps=0

    ## If do_extrapolation, energies calculated wit num cnot pairs for circuit fold 
    ## ranging from 0 to max number and extrapolation carried out between time steps. 
    if do_extrapolation == True:
        if number_circuit_folds>0:
            extrapolation="circuit_folding"
            max_number_circuit_folds=number_circuit_folds
            max_number_cnot_pairs=0
            
        elif number_cnot_pairs>0:
            extrapolation="cnot_pairs"
            max_number_cnot_pairs=number_cnot_pairs
            max_number_circuit_folds=0
        else:
            raise ValueError("either number_circuit_folds or number_cnot_pairs must be non-zero")
        
        num_extrapolation_steps=max_number_circuit_folds+max_number_cnot_pairs
        number_cnot_pairs=0
        number_circuit_folds=0
        print(f"Extrapolating with {extrapolation}")

    ##otherwise, number_cnot_pairs or number_circuit_folds simply carried through
    else:
        ## only run the circuit once
        num_extrapolation_steps=0


    ## Get number of qubits 
    n_qubits=H.N_qubits

    ## Get list of sigmas (all pauli terms with odd number Y gates)
    sigmas=sigma_terms(n_qubits,encoding)

    ## Construct b and S in terms of paulis form sigmas 
    b_pauli_terms=b_terms(H,sigmas)
    S_pauli_terms=S_terms(sigmas)

    ## Get commuting set of pauli terms 
    pauli_set=get_intersection_pauli_terms(H,b_pauli_terms,S_pauli_terms)
    commuting_sets=get_commuting_sets(sorted(pauli_set))
    
    ## Zero initialize containers
    evolution_set=[]
    a=np.zeros(len(sigmas))
    Energies=np.zeros(time_steps)
    Ccoefs=np.zeros(time_steps)

    ## Construct noise model
    noise_model=None
    device=None
    if parameters['device_name'] is not None:
        if ( parameters['device_name'][0:6] == "custom" ):
            errors=[float(i) for i in parameters['device_name'][6:].split("_")]
            noise_model=create_custom_noise_model(errors)
        else:
            device = Device(parameters['device_name'], parameters['mitigate_meas_error'], 
                            H.N_qubits, layout=parameters['layout'])
            
    ## TODO:  Convert to print to logfile 
    if verbose:
        if device is not None:
            print("Device specifications")
            pprint(vars(device))

    ## for each time step, run circuit and compute A for the next time step
    ## If do extrapolation set to true, extrapolate energies after each time step
    if show_progress:
        time_range=tqdm(range(time_steps))
    else:
        time_range=range(time_steps)

    for t in time_range:
        energies_for_extrapolation=[]

        
        for num in range(num_extrapolation_steps+1):
            if extrapolation == "circuit_folding":
                number_circuit_folds=num 
            ## If extrapolation = cnot_pairs or None, set number_cnot_pairs to num
            ## If extrapolation is None, number_cnot_pairs is just set to zero. 
            else:
                number_cnot_pairs=num

            expectation_values={}
            if backend=='statevector_simulator':
                ## Run circuit to get state vector
                psi=run_circuit_statevector(
                        n_qubits,evolution_set,delta_time,
                        initialization=initialization,
                        encoding=encoding,
                        number_cnot_pairs=number_cnot_pairs,
                        number_circuit_folds=number_circuit_folds
                        )

                ## Compute expectation value for each pauli term 
                for pauli_id in commuting_sets:        
                    for pauli in commuting_sets[pauli_id]: 
                        pauli_mat = get_pauli_matrix(pauli)
                        e_value=np.conj(psi).T @ pauli_mat @ psi
                        expectation_values[pauli]=e_value

            else:
                for pauli_id in commuting_sets:   
                    ## Run circuit to get counts 
                    meas_results=run_circuit_qasm(
                                    n_qubits,evolution_set,
                                    pauli_id,delta_time,
                                    n_shots=n_shots,
                                    initialization=initialization,
                                    encoding=encoding,
                                    noise_model=noise_model,
                                    device=device,
                                    number_cnot_pairs=number_cnot_pairs,
                                    number_circuit_folds=number_circuit_folds
                                    )

                    ## Compute expectation value for each pauli term 
                    for pauli in commuting_sets[pauli_id]: 
                        expectation_values[pauli]=compute_expectation_value(pauli,meas_results)    


            ## Compute energy
            H_pauli=H.pauli_coeffs
            energy=0.0
            for key in H_pauli:
                energy+=H_pauli[key]*expectation_values[key]

            energies_for_extrapolation.append(energy.real)
        
        ## If extrapolation == None then circuit only run once with num=0
        ## Don't do extrapolation if t==0
        if (extrapolation == None) or (t==0):
            Energies[t]=energies_for_extrapolation[0]
        
        ## Otherwise, do extraplation first
        else:
            ## Do extrapolation
            x=[2*n+1 for n in range(num_extrapolation_steps+1)]
            ##TODO make polynomial degrees a function
            poly_degree=parameters['degree_extrapolation_polynomial']
            coef=np.polyfit(x,energies_for_extrapolation,poly_degree)
            fit=np.poly1d(coef)
            ## Extrapolated energy is y intercept at x=0
            extrapolated_energy=fit(0)
            # print(energies_for_extrapolation)
            Energies[t]=extrapolated_energy
            # print(Energies[t])
        ## compute normalization coef C=1-2*E*delta_times
        Ccoef=1-2*delta_time*Energies[t]
        Ccoefs[t]=Ccoef

        ## Compute A
        evolution_set.append(A_pauli_operator(delta_time,sigmas,S_pauli_terms,b_pauli_terms,expectation_values,Ccoef,A_threshold))

        ## Merge evolution operators if time step is a merge step
        if isinstance(merge_step,int):
            if t%merge_step==0:
                identity_string="I"*n_qubits
                A_combine=WeightedPauliOperator([(0.0,Pauli.from_label(identity_string))])
                for A in evolution_set:
                    A_combine+=A
                evolution_set=[A_combine]

    ## Just return calculated energies
    if QLanczos:
        return Energies, Ccoefs

    ## Otherwise, do QLanczos calculation
    else:
        return Energies

###########################################################################################


def do_qite_experiment(trial_index,parameters,show_progress=True):
    ## Construct Hamiltonian in HO basis
    H = hamiltonian_matrix(Nmax=parameters['Nmax'], J=1, interaction=parameters['interaction'])

    ## Construct qubit Hamiltonian
    if parameters['encoding'] == 'gray_code':
        H_qubit = GrayCodeHamiltonian(H)
        
    elif parameters['encoding'] == 'jordan_wigner':
        H_qubit = JordanWignerHamiltonian(H)

    if(trial_index==0):
        print("Hamiltonian Pauli rep")
        print(H_qubit.pauli_rep,"\n")
        print("Pauli partitions")
        print(H_qubit.pauli_partitions,"\n")

    results=qite_experiment2(H_qubit,parameters)

    # print(energies)
    return results

def do_qlanczos_experiment(trial_index,parameters):
    ## Construct Hamiltonian in HO basis
    assert(parameters['qlanczos']==True)

    H = hamiltonian_matrix(Nmax=parameters['Nmax'], J=1, interaction=parameters['interaction'])

    ## Construct qubit Hamiltonian
    if parameters['encoding'] == 'gray_code':
        H_qubit = GrayCodeHamiltonian(H)
        
    elif parameters['encoding'] == 'jordan_wigner':
        H_qubit = JordanWignerHamiltonian(H)

    if(trial_index==0):
        print("Hamiltonian Pauli rep")
        print(H_qubit.pauli_rep,"\n")
        print("Pauli partitions")
        print(H_qubit.pauli_partitions,"\n")

    Energies,Ccoefs=qite_experiment2(H_qubit,parameters)

    ## Do QLanczos on computed vectors
    krylov_threshold = parameters['krylov_threshold']
    qlanczos_threshold=parameters['qlanczos_threshold']
    qlanczos_energy=do_QLanczos(
        Ccoefs,Energies,
        krylov_threshold=krylov_threshold,
        zero_threshold=qlanczos_threshold,
        reverse_order=True
        )

    return qlanczos_energy
   
def run_qlanczos_experiment_iterative(parameters,krylov_dim):
    """
    Run qite evolution to get energies of ground state 
    """
    Nmax=parameters['Nmax']
    interaction=parameters['interaction']
    H = hamiltonian_matrix(Nmax=Nmax,J=1,interaction=interaction)
    H_qubit = GrayCodeHamiltonian(H)
    
    delta_time = parameters['delta_time']
    backend = parameters['backend']
    initialization = parameters['initialization']
    max_iterations=parameters['N_time_steps']
    n_shots=parameters['N_shots']
    qite_threshold=parameters['qite_threshold']
    merge_step=parameters['merge_step']
    n_qubits=H_qubit.N_qubits
    krylov_threshold=parameters['krylov_threshold']
    ## Get list of sigmas (all pauli terms with odd number Y gates)
    sigmas=sigma_terms(n_qubits)
    ## Construct b in terms of paulis 
    b_pauli_terms=b_terms(H_qubit,sigmas)
    
    ## Construct S in terms of paulis
    S_pauli_terms=S_terms(sigmas)

    ## Get composite set of pauli terms that need to be calculated for QITE 
    pauli_set=get_intersection_pauli_terms(H_qubit,b_pauli_terms,S_pauli_terms)

    ## Get commuting set 
    commuting_sets=get_commuting_sets(sorted(pauli_set))
    
    ## Zero initialize 
    evolution_set=[]
    a=np.zeros(len(sigmas))
    Energies=[]
    Ccoefs=[]
    ncoefs=[]
    
    ## for each time step, run circuit and compute A for the next time step
    krylov_indices=[0]
    t=0
    while len(krylov_indices)<krylov_dim:
        if t>max_iterations:
            break
        expectation_values={}
        if backend=='statevector_simulator':
            ## Run circuit to get state vector
            psi=run_circuit_statevector(n_qubits,evolution_set,delta_time,initialization=None)

            ## Compute expectation value for each pauli term 
            for pauli_id in commuting_sets:        
                for pauli in commuting_sets[pauli_id]: 
                    pauli_mat = get_pauli_matrix(pauli)
                    e_value=np.conj(psi).T @ pauli_mat @ psi
                    expectation_values[pauli]=e_value

        else:
            for pauli_id in commuting_sets:   
                ## Run circuit to get counts 
                meas_results=run_circuit_qasm(n_qubits,evolution_set,pauli_id,delta_time,n_shots=n_shots,initialization=None)

                ## Compute expectation value for each pauli term 
                for pauli in commuting_sets[pauli_id]: 
                    expectation_values[pauli]=compute_expectation_value(pauli,meas_results)    


        ## Compute energy
        H_pauli=H_qubit.pauli_coeffs
        energy=0.0
        for key in H_pauli:
            energy+=H_pauli[key]*expectation_values[key]

        Energies.append(energy.real)

        ## compute normalization coef C=1-2*E*delta_times
        Ccoef=1-2*delta_time*Energies[t]
        Ccoefs.append(Ccoef)
        
        ## Compute A
        evolution_set.append(A_pauli_operator(delta_time,sigmas,S_pauli_terms,b_pauli_terms,expectation_values,Ccoef,qite_threshold))

        if isinstance(merge_step,int):
            if t%merge_step==0:
                identity_string="I"*n_qubits
                A_combine=WeightedPauliOperator([(0.0,Pauli.from_label(identity_string))])
                for A in evolution_set:
                    A_combine+=A
                evolution_set=[A_combine]
         
        ### Compute normalization factor 
        if t==0:
            ncoefs.append(1.0)
        else:
            ncoefs.append(normalization_coefficient(ncoefs,Ccoefs,t))
            
            ## If even time step, check overlap
            if t%2==0:        
                i=t
                j=krylov_indices[-1]
                k=int((i+j)/2)
                regularization_factor=(ncoefs[i]*ncoefs[j])/(ncoefs[k]**2)
                if regularization_factor<krylov_threshold:
                    krylov_indices.append(i)
        
        ### increment t
        t+=1    

    ## Construct krylove matries 
    threshold=1e-2
    T_krylov,H_krylov=Krylov_matrices(krylov_indices,ncoefs,Energies,zero_threshold=threshold)

    ## Solve generalized eigenproblem
    eigs,vecs=eig(H_krylov,T_krylov)
    print(f"Iterations: {t-1}/{max_iterations}")
    print(f"Krylov dim: {len(krylov_indices)}")
    return eigs.real




def analyze_qite_results(results_filename):
    ## Each row corresponds to a given time step
    results=np.load(results_filename)
    averages=np.mean(results,0)
    stdevs=np.std(results,0)
    # print(averages)
    # print(stdevs)
    return averages,stdevs



