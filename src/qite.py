import numpy as np 
from scipy.linalg import lstsq

from tqdm import tqdm

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute, Aer
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.initial_states import Custom

from hamiltonian import *
from utils import *
from qiskit_circuits import *

import itertools

def commutes(a,b):
    """
    Returns true if two operators commute
    
    input:
        a,b (str) : Pauli operators.  Must be "X", "Y", "Z", "I"
    """
    #Todo add raise error if a[i] not in allowed pauli set
    does_commute=True
    for i in range(len(a)):
        ## If ai or bi are identity then ith paulis will commute 
        if a[i]=='I' or b[i]=='I':
            continue
        
        ## if ai != bi then they won't commute 
        if a[i]!=b[i]:
            does_commute=False
            break

    return does_commute


def get_pauli_terms(weighted_pauli):
    """
    Extract out individual paul terms 
    
    Input:
        weighted_pauli ():
        
    Return
        terms (dictionary) : dictionary of {label:coef} for each pauli term. 
                label is pauli string
    """
    terms={}
    for term in weighted_pauli.to_dict()['paulis']:
        label=term['label']
        coef=term['coeff']['real']+1j*term['coeff']['imag']
        terms[label]=coef
    return terms


def get_sigma_pauli_terms(n_qubits):
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
            paulis.append(Pauli.from_label(string))

    sigmas=[]
    for pauli in paulis:
        sigma=[(1.0,pauli)]
        sigmas.append(WeightedPauliOperator(sigma))

    return sigmas


def H_weighted_paulis(H):
    """
    Converts Hamiltonian operator into a WeightedPauliOperator

    Input: 
        H (hamiltonian.EncodingHamiltonian) : Qubit Hamiltonian

    Returns (WeightedPauliOperator) : Qubit Hamiltonian expressed as a WeightedPauliOperator

    """
    H_pairs=[(v, Pauli.from_label(k)) for (k, v) in list(H.pauli_coeffs.items())]
    return WeightedPauliOperator(H_pairs)   
    
    
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
    H_paulis=H_weighted_paulis(H)
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

def compute_expectation_value(pauli,meas_results):
    """
    Compute expectation value of pauli operator using counts obtained by running qiskit circuit

    """
    pauli_list = list(pauli)
    n_qubits = len(pauli_list)
    expectation_value=0.0
    for basis_state in meas_results.keys():
        num_z_and_1 = [-1 if (basis_state[bit_idx] == '1' and pauli_list[bit_idx] != 'I') else 1 for bit_idx in range(n_qubits)]
        eigenvalue = reduce(lambda x, y: x*y, num_z_and_1)
        expectation_value+=eigenvalue*meas_results[basis_state]

    return expectation_value

def get_commuting_sets(paulis):
    """
    Get a dictionary of commuting sets.  
    
    Key for each set is term with fewest number of idenity operators 
    and thus provides key for measurement basis
    
    input:
        paulis (list<str>) : list of pauli terms
    
        
    """
    commuting_array=[]
    paulis=list(paulis)
    for p1 in paulis:
        does_commute=False
        for i in range(len(commuting_array)):
            p_set=commuting_array[i]
            for p2 in p_set:
                does_commute=commutes(p1,p2)
                if not does_commute:
                    break
            if does_commute:
                commuting_array[i].append(p1)
                break
        if not does_commute:
            commuting_array.append([p1])

    ## identify pauli which will define measurement basis
    commuting_sets={}
    n_qubits=len(paulis[0])
    for p_set in commuting_array:
        minI=n_qubits
        identitfier=None
        for p in p_set:
            numI=p.count("I")
            if numI < minI:
                minI=numI
                identifier=p
        commuting_sets[identifier]=p_set
    return commuting_sets

def initialize_circuit(q,c,initial_state="zeros"):
    """
    Initialize circuit.  
    
    input:
        q (qiskit.circuit.quantumregister.QuantumRegister) : qubits
        initial_state(str or np.array) : initial state of circuit
            "zeros" (str,default) : starting state is qubit state with all qubits in zero state
            "uniform" (str) : Uniform superposition of all qubit states 
             array(np.array) : normalized array of len(2**n_qubits)
    returns
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : initialized quantum circuit
    """    

    if initial_state is None:
        initial_state="zeros"
        
    if isinstance(initial_state, str):
        if initial_state=="zeros":
            circuit = QuantumCircuit(q,c)

        elif initial_state=="uniform":
            circuit = QuantumCircuit(q,c)
            circuit.h(q)
        else:

            print(initial_state, "not currently valid option")
            sys.exit(f"{initial_state} not currently valid option")
    else:
        circuit = QuantumCircuit(q,c)
        circuit.initialize(initial_state,q)
    return circuit


def get_evolution_circuit(q,A_set,circuit):
    """
    Append evolution exp(-iAt) onto circuit for each time step t
    
    input:
        q (qiskit.circuit.quantumregister.QuantumRegister) : qubits
        A_set () : List of Weighted Pauli operators by which the circuit is evolved at each timestep
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : quantum circuit 

    return
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : updated quantum circuit
    """

    for A in A_set: 
        #Append next step to circuit for each A
        circuit += A.evolve(
            None, evo_time=1, num_time_slices=1,
            quantum_registers=q
            )
    return circuit  

def get_measurement_circuit(q,c,circuit,set_id):
    """
    Append gates to transform qubit states to measurement basis 
    
    input 
        q (qiskit.circuit.quantumregister.QuantumRegister) : qubits
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : quantum circuit 
        set_id (str) : pauli operator to be measured
    return
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : updated quantum circuit        
    """
    ## Reverse order to gates are applied right to left
    id=set_id[::-1]
    
    for qubit_idx in range(len(set_id)):
        pauli=id[qubit_idx]
        if pauli == 'X':
            circuit.h(q[qubit_idx])
        elif pauli == 'Y':
            circuit.sdg(q[qubit_idx])
            circuit.h(q[qubit_idx])
    
    circuit.measure(q,c)
    
    return circuit


def run_circuit_statevector(n_qubits,A_set,initialization=None):
        ## Initalize circuit
#         print("qubits ", n_qubits)
        q = QuantumRegister(n_qubits)
        c = ClassicalRegister(n_qubits)
        
        circuit=initialize_circuit(q,c,initial_state=initialization)

        ## If A_set not t, then evolve cirucit using previously computed A matrices stored in A_set
        if len(A_set)>0:
            circuit=get_evolution_circuit(q,A_set,circuit)

        ## Execute circuit
        job = execute(circuit, Aer.get_backend("statevector_simulator"))
        
        return job.result().get_statevector(circuit)        

    
def run_circuit_qasm(n_qubits,A_set,pauli_id,n_shots=1024,initialization=None):
        ## Initalize circuit
        q = QuantumRegister(n_qubits)
        c = ClassicalRegister(n_qubits)
        
        circuit=initialize_circuit(q,c,initial_state=initialization)

        ## If A_set not t, then evolve cirucit using previously computed A matrices stored in A_set
        if len(A_set)>0:
            circuit=get_evolution_circuit(q,A_set,circuit)
        
        ## Rotate to measurement basis 
        circuit=get_measurement_circuit(q,c,circuit,pauli_id)       
        
        ## Execute circuit
        job = execute(circuit, Aer.get_backend("qasm_simulator"),shots=n_shots)
        
        ## Get counts
        counts= job.result().get_counts(circuit)

        ##normalize counts
        for state in counts:
            counts[state]=counts[state]/n_shots
        
        return counts

def A_pauli_operator(delta_time,sigmas,S_pauli_terms,b_pauli_terms,expectation_values,Ccoef,A_threshold,verbose=False):
        
    
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
        A+=delta_time*a[i]*sigmas[i]

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

def run_qite_experiment(H,num_iterations,delta_time,backend,initialization,A_threshold=1e-10,cstep=None):
    """
    Run qite evolution to get energies of ground state 
    """

    n_qubits=H.N_qubits
#     print("num qubits",n_qubits)
    n_shots=10000 ## Set to allowed number of shots at IBMQ

    ## Get list of sigmas (all pauli terms with odd number Y gates)
    sigmas=get_sigma_pauli_terms(n_qubits)

    ## Construct b in terms of paulis 
    b_pauli_terms=b_terms(H,sigmas)
    
    ## Construct S in terms of paulis
    S_pauli_terms=S_terms(sigmas)

    ## Get composite set of pauli terms that need to be calculated for QITE 
    pauli_set=get_intersection_pauli_terms(H,b_pauli_terms,S_pauli_terms)

    ## Get commuting set 
    commuting_sets=get_commuting_sets(sorted(pauli_set))
    
    ## Zero initialize 
    A_set=[]
    a=np.zeros(len(sigmas))
    Energies=np.zeros(num_iterations)
    Ccoefs=np.zeros(num_iterations)
    ## for each time step, run circuit and compute A for the next time step
    for t in tqdm(range(num_iterations)):
#     for t in range(num_iterations):
    #     print("")
        expectation_values={}
        if backend=='statevector_simulator':
            ## Run circuit to get state vector
            psi=run_circuit_statevector(n_qubits,A_set,initialization=None)

            ## Compute expectation value for each pauli term 
            for pauli_id in commuting_sets:        
                for pauli in commuting_sets[pauli_id]: 
                    pauli_mat = get_pauli_matrix(pauli)
                    e_value=np.conj(psi).T @ pauli_mat @ psi
                    expectation_values[pauli]=e_value

        else:
            for pauli_id in commuting_sets:   
                ## Run circuit to get counts 
                meas_results=run_circuit_qasm(n_qubits,A_set,pauli_id,n_shots=n_shots,initialization=initialization)

                ## Compute expectation value for each pauli term 
                for pauli in commuting_sets[pauli_id]: 
                    expectation_values[pauli]=compute_expectation_value(pauli,meas_results)    


        ## Compute energy
        H_pauli=H.pauli_coeffs
        for key in H_pauli:
            Energies[t]+=H_pauli[key]*expectation_values[key]

        ## compute normalization coef C=1-2*E*delta_times
        Ccoef=1-2*delta_time*Energies[t]
        Ccoefs[t]=Ccoef
        ## Compute A
        A_set.append(A_pauli_operator(delta_time,sigmas,S_pauli_terms,b_pauli_terms,expectation_values,Ccoef,A_threshold))

        if isinstance(cstep,int):
            if t%cstep==0:
                identity_string="I"*n_qubits
                A_combine=WeightedPauliOperator([(0.0,Pauli.from_label(identity_string))])
                for A in A_set:
                    A_combine+=A
                A_set=[A_combine]

    return Energies,Ccoefs

