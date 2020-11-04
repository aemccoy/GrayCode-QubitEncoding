import numpy as np 
import pytest

import sys
sys.path.append("./src/")
from hamiltonian import *

# Generate "true" ground state energies of the deuteron Hamiltonian
# Our Hamiltonians should be able to reproduce all these values.

def test_energies(interaction_filename,hw,J,encoding,Nmax_max):
    ## If qiskit ordering False
    for Nmax in range(0,Nmax_max+1,2):
        H = hamiltonian_matrix(Nmax=Nmax,hw=hw,J=J,interaction_filename=interaction_filename) 
        exact_energy=np.linalg.eigh(H)[0][0]

        if encoding=="Graycode":
            H_qubit = GrayCodeHamiltonian(H,qiskit_order=False)
        elif encoding =="JordanWigner":
            H_qubit = JordanWignerHamiltonian(H,qiskit_order=False)
        else:
            raise ValueError(f"Encoding {encoding} is not valid type.\nPlease choose 'Graycode' or 'JordanWigner'")

        energy = np.min(np.linalg.eigh(H_qubit.matrix)[0])
        assert np.isclose(energy, exact_energy)

    ## If qiskit ordering True
    for Nmax in range(0,Nmax_max+1,2):
        H = hamiltonian_matrix(Nmax=Nmax,hw=7.0,J=1,interaction_filename="toy_hamiltonian") 
        
        exact_energy=np.linalg.eigh(H)[0][0]
        
        if encoding=="Graycode":
            H_qubit = GrayCodeHamiltonian(H,qiskit_order=True)
        elif encoding =="JordanWigner":
            H_qubit = JordanWignerHamiltonian(H,qiskit_order=True)
        else:
            raise ValueError(f"Encoding {encoding} is not valid type.\nPlease choose 'Graycode' or 'JordanWigner'")

        energy = np.min(np.linalg.eigh(H_qubit.matrix)[0])
        assert np.isclose(energy, exact_energy)

    print("Energy test complete")

def inspect_graycode_hamiltonian(interaction_filename,hw,J,Nmax,qiskit_order=True):
    H = hamiltonian_matrix(Nmax=Nmax,hw=hw,J=J,interaction_filename=interaction_filename)
    H_qubit = GrayCodeHamiltonian(H,qiskit_order=False)
    print("GrayCodeHamiltonian")
    print(f"N_states: {H_qubit.N_states}")
    print(f"N_qubits: {H_qubit.N_qubits}\n")
    print("Gray code states:\n", f"{H_qubit.state_order}")   
    print("Fermionic representation")
    print(H_qubit.ferm_rep)
    print("")
    print(f"Number of Pauli partitions: {H_qubit.n_partitions}")
    print("Pauli partitions")
    print(H_qubit.pauli_partitions)
    print("")
    print("Coefficient on each Pauli term")
    print(H_qubit.pauli_coeffs)
    print("")
    print("H as weighted Pauli operator")
    print(H_qubit.weighted_pauli.print_details())
    print("")
    print("H as matrix")
    print(f"qiskit order: {H_qubit.qiskit_order}")
    print(H_qubit.matrix)

    
def inspect_jordanwigner_hamiltonian(interaction_filename,hw,J,Nmax,qiskit_order=True):
    H = hamiltonian_matrix(Nmax=Nmax,hw=hw,J=J,interaction_filename=interaction_filename)
    H_qubit = JordanWignerHamiltonian(H,qiskit_order=False)
    print("GrayCodeHamiltonian")
    print(f"N_states: {H_qubit.N_states}")
    print(f"N_qubits: {H_qubit.N_qubits}\n")
    print("Fermionic representation")
    print(H_qubit.ferm_rep)
    print("")
    print(f"Number of Pauli partitions: {H_qubit.n_partitions}")
    print("Pauli partitions")
    print(H_qubit.pauli_partitions)
    print("")
    print("Coefficient on each Pauli term")
    print(H_qubit.pauli_coeffs)
    print("")
    print("H as weighted Pauli operator")
    print(H_qubit.weighted_pauli.print_details())
    print("")
    print("H as matrix")
    print(f"qiskit order: {H_qubit.qiskit_order}")
    print(H_qubit.matrix)



if(__name__ == "__main__"):
    # test_energies("toy_hamiltonian",7.0,1,"JordanWigner",16)
    # test_energies("toy_hamiltonian",7.0,1,"Graycode",16)
    
    # inspect_graycode_hamiltonian("toy_hamiltonian",hw=7.0,J=1,Nmax=4,qiskit_order=True)
    # inspect_jordanwigner_hamiltonian("toy_hamiltonian",hw=7.0,J=1,Nmax=4,qiskit_order=True)
    
    filename="interactions/Heff_postsrg_3S1-3D1_n4lo500-srg1.5_18_14.dat"
    basis,H_matrix=get_hamiltonian(filename)
    print(H_matrix)