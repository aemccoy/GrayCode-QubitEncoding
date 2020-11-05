import numpy as np 
import pytest

import sys
sys.path.append("./src/")
from hamiltonian import *

# Generate "true" ground state energies of the deuteron Hamiltonian
# Our Hamiltonians should be able to reproduce all these values.

def test_energies(Nmax_max,J,interaction,encoding):
    ## If qiskit ordering False
    for Nmax in range(0,Nmax_max+1,2):
        H = hamiltonian_matrix(Nmax,J,interaction) 
        exact_energy=np.linalg.eigh(H)[0][0]

        if encoding=="gray_code":
            H_qubit = GrayCodeHamiltonian(H,qiskit_order=False)
        elif encoding =="jordan_wigner":
            H_qubit = JordanWignerHamiltonian(H,qiskit_order=False)
        else:
            raise ValueError(f"Encoding {encoding} is not valid type.\nPlease choose 'Graycode' or 'JordanWigner'")

        energy = np.min(np.linalg.eigh(H_qubit.matrix)[0])
        assert np.isclose(energy, exact_energy)

    ## If qiskit ordering True
    for Nmax in range(0,Nmax_max+1,2):
        H = hamiltonian_matrix(Nmax,J,interaction)
        
        exact_energy=np.linalg.eigh(H)[0][0]
        
        if encoding=="gray_code":
            H_qubit = GrayCodeHamiltonian(H,qiskit_order=True)
        elif encoding =="jordan_wigner":
            H_qubit = JordanWignerHamiltonian(H,qiskit_order=True)
        else:
            raise ValueError(f"Encoding {encoding} is not valid type.\nPlease choose 'Graycode' or 'JordanWigner'")

        energy = np.min(np.linalg.eigh(H_qubit.matrix)[0])
        assert np.isclose(energy, exact_energy)

    print("Energy test complete")

def inspect_graycode_hamiltonian(Nmax,J,interaction,qiskit_order=True):
    H = hamiltonian_matrix(Nmax=Nmax,J=J,interaction=interaction)
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

    
def inspect_jordanwigner_hamiltonian(Nmax,J,interaction,qiskit_order=True):
    H = hamiltonian_matrix(Nmax=Nmax,J=J,interaction=interaction)
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

def TestHamiltonianConstruction():
    Nmax=18
    J=1

    for interaction in ["toy","N4LOsrg1.5","Daejeon","N3L0srg2.15","N2LOopt"]:    
        H_matrix=hamiltonian_matrix(Nmax,J,interaction)
        eigs,vecs=np.linalg.eigh(H_matrix)
        print(f"{interaction}: {eigs[0]}")


if(__name__ == "__main__"):
    test_energies(Nmax_max=16,J=1,interaction="toy",encoding="jordan_wigner")
    test_energies(Nmax_max=16,J=1,interaction="toy",encoding="gray_code")
    
    inspect_graycode_hamiltonian(Nmax=4,J=1,interaction="toy",qiskit_order=True)
    inspect_jordanwigner_hamiltonian(Nmax=4,J=1,interaction="toy",qiskit_order=True)
    
    TestHamiltonianConstruction()
