import numpy as np 
import pytest

import sys
sys.path.append("./src/")
from hamiltonian import *
from utils import *
from qiskit_circuits import *
# Generate "true" ground state energies of the deuteron Hamiltonian
# Our Hamiltonians should be able to reproduce all these values.


if(__name__ == "__main__"):

    ##testing state initialization 
    if False:    
        q = QuantumRegister(n_qubits)
        c = ClassicalRegister(n_qubits)
    #     init=np.random.uniform(0,1,2**n_qubits)
    #     init=init/np.sqrt(init.dot(init))

        print("initialization vector",init)
        print(init.dot(init))

        circuit = initialize_circuit(q,c,initial_state=init)
        backend = 'statevector_simulator'
        
        job = execute(circuit, Aer.get_backend(backend),shots=n_shots)
        
        state_vector=job.result().get_statevector(circuit)
        counts = job.result().get_counts(circuit)
        n_shots = sum(counts.values())
        print("state vector : ",state_vector)
        print("num shots :",n_shots)
        print("counts :",counts)
        print(sum(counts.values()))
        
        print()

    A_set=[]
    n_qubits=2
    pauli_id="I"*n_qubits

    state_vector=run_circuit_statevector(n_qubits,A_set,initialization=None)


    counts=run_circuit_qasm(n_qubits,A_set,pauli_id,n_shots=10000,initialization=None)
    vec=[]
    for state in counts:
        vec.append(np.sqrt(counts[state]))
        
    print(state_vector)
    print(counts)
    print(vec)    
