from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

import numpy as np

def add_cnot_gate(circuit,q,c,t,num_cnot_pairs=0):
    """
    Add (num_cnot_pairs+1) CNOT gates to ith and jth qubits [j=(i+1)%N_qubits] 
    to circuit and append the CNOT gates to operators
    
    Inputs:
        circuit (QuantumCircuit) : quantum circuit
        
        q (QuantumRegister) : quantum register for circuit
        
        c (int) : index of control qubit
        t (int) : index of target qubit 
        
        num_cnot_pairs (int,optional) : number of pairs of CNOT gates inserted as 
            "resolution of identity" with every CNOT gate in circuit.  
            Used for error extrapolation.
            
    Returns:
        circuit with CNOT gates appended
    """
    for n_pairs in range(num_cnot_pairs):
        circuit.cx(q[c], q[t])
        circuit.cx(q[c], q[t])

    circuit.cx(q[c], q[t])
    
    return circuit

def fold_circuit(circuit,num_folding):
    """
    Applies and then inverts circuit num_folding times before finally running the circuit
    Used for error extrapolation? 
    
    Input:
        circuit (QuantumCircuit) : quantum circuit 
        
        num_folding (int) : Number of times circuit is "folded"
        
    Returns:
        circuit (QuantumCircuit) : folded circuit
    
    """
    folded_circuit=circuit
    for n in range(num_folding):
        folded_circuit=folded_circuit.combine(circuit.inverse())
        folded_circuit=folded_circuit.combine(circuit)
    return folded_circuit


def append_Gray_code_variational_ansatz(circuit,q,thetas,num_cnot_pairs=0):
    """ 
    Creates a variational ansatz for the Gray code encoding.

    These circuits have a more familiar 'variational form'. Since in the dense case we
    are using all elements in the space, we can use something a bit more general. These
    circuits are layers of Y rotations followed by entangling gates on all the qubits.

    Parameters:
        circuit (QuantumCircuit) : quantum circuit
        
        q (QuantumRegister) : quantum register for circuit
        
        thetas (np.array) : Angles parameterizing ansatz wavefunction.  Number of
            angles is one fewer than the number of states.
        
        num_cnot_pairs (int,optional) : number of pairs of CNOT gates inserted as 
            "resolution of identity" with every CNOT gate in circuit.  
            Used for error extrapolation.
            
    Returns:
        circuit (QuantumCircuit) : circuit with variational ansatz added

    """
    N_qubits = q.size

    num_remaining_thetas = len(thetas)
    qubit_idx = 0
    last_layer = False

    while num_remaining_thetas > 0:
        # If it's the last layer, only do as many CNOTs as we need to
        if last_layer:
            for i in range(num_remaining_thetas):
                t=(i+1)%N_qubits ## target
                circuit=add_cnot_gate(circuit,q,i,t,num_cnot_pairs)
         
            for qubit in range(1, num_remaining_thetas+1):
                circuit.ry(thetas[num_remaining_thetas - 1], q[qubit%N_qubits])
                num_remaining_thetas -= 1
            break
        # If it's not the last layer, apply rotations and then increment counters
        else:
            circuit.ry(thetas[num_remaining_thetas - 1], q[qubit_idx])
            qubit_idx += 1
            num_remaining_thetas -= 1

            if N_qubits >= 2:
                # If we have a full layer of parameters ahead, do the full CNOT cycle
                if qubit_idx == N_qubits and num_remaining_thetas > N_qubits:
                    for i in range(N_qubits):
                        t=(i+1)%N_qubits ## target
                        operations=add_cnot_gate(circuit,q,i,t,num_cnot_pairs)
                    ##Reset qubit_idx to zero.  
                    qubit_idx = 0
                
                # Otherwise, we are entering the last layer
                elif qubit_idx == N_qubits and num_remaining_thetas <= N_qubits:
                    last_layer = True
                
    return circuit

def append_Jordan_Wigner_variational_ansatz(circuit, q, thetas, num_cnot_pairs=0):
    """ Creates a variational ansatz for the Jordan-Wigner encoding.

    These circuits were defined in arXiv:1904.04338, and produce an ansatz state
    over the occupation subset of the computational basis (|1000>, |0100>, etc.)
    with generalized spherical coordinates as their amplitudes.

    The version here is the same circuit, but with the qubit order inverted.
    This was found to lead to greater stability in the SPSA optimization procedure
    due to the relationships between of the variational parameters and the
    "strength" of the basis states in the Hamiltonian.

    Parameters:
        thetas (np.array) : Angles parameterizing ansatz wavefunction.  Number of
            angles is one fewer than the number of qubits in the circuit.

    Returns:
        circuit (QuantumCircuit) : quantum circuit

    """
    # The number of parameters tells us the number of states 
    # which is the same as the the number of qubits 
    N_states = len(thetas) + 1
    N_qubits = N_states
 
    circuit.x(q[N_qubits-1])
    circuit.ry(thetas[0], q[N_qubits-2])

    # Insert CNOT gates
    control_idx,target_idx=N_qubits-2,N_qubits-1
    circuit=add_cnot_gate(circuit,q,control_idx,target_idx,num_cnot_pairs)

    # Recursive cascade
    for control_idx in range(N_qubits - 2, 0, -1):
        target_idx = control_idx - 1

        circuit.ry(thetas[control_idx]/2, q[target_idx])        
        circuit=add_cnot_gate(circuit,q,control_idx,target_idx,num_cnot_pairs)        
        circuit.ry(-thetas[control_idx]/2, q[target_idx])
        circuit=add_cnot_gate(circuit,q,control_idx,target_idx,num_cnot_pairs)
        circuit=add_cnot_gate(circuit,q,target_idx,control_idx,num_cnot_pairs)
    
    return circuit


def append_measurement_basis_rotation(circuit,q,measurement_id):
    """
    Append gates to transform qubit states to measurement basis 
    
    input:
        circuit (QuantumCircuit) : quantum circuit 

        q (qiskit.circuit.quantumregister.QuantumRegister) : qubits
        
        measurement_id (str) : pauli operator to be measured
    
    returns:
        circuit (QuantumCircuit) : circuit now transformed to 
                measurement basis defined by measurement_id
    """
    ## Reverse order to gates are applied right to left Maybe??
    measurement_id_revered=measurement_id[::-1]

    for qubit_idx in range(len(measurement_id)):
#         print(f"qubit {qubit_idx} of {measurement_id}")
        pauli=measurement_id_revered[qubit_idx]
        if pauli == 'X':
            circuit.h(q[qubit_idx])
        elif pauli == 'Y':
            circuit.sdg(q[qubit_idx])
            circuit.h(q[qubit_idx])

    return circuit


def variational_circuit(encoding,thetas,measurement_idx,backend_name,num_cnot_pairs=0,num_folding=0):
    """
    Construct variational circuit 

    input:
        encoding (string) : string identifier for which type of variational circuit to run 
                "Graycode" : variational ansatz for dense Gray code encoding
                "JordanWigner" : variational ansatz for Jordan Wigner encoding

    returns:
        circuit (QuantumCircuit) : variational circuit
    """
    N_states=len(thetas)+1
    if encoding=="Graycode":
        N_qubits = int(np.ceil(np.log2(N_states)))
    elif encoding == "JordanWigner":
        N_qubits=N_states
    else:
        raise ValueError("Invalid encoding for variational circuit.")
        
    q, c = QuantumRegister(N_qubits), ClassicalRegister(N_qubits)
    circuit = QuantumCircuit(q, c)
    
    if encoding=="Graycode":
        circuit=append_Gray_code_variational_ansatz(circuit,q,thetas,num_cnot_pairs)
    else: # must be "JordanWigner" else exception raised earlier 
        circuit=append_Jordan_Wigner_variational_ansatz(circuit,q,thetas,num_cnot_pairs)
        
    if backend_name=="qasm_simulator":
#         print(f"m idx: {measurement_idx}")
        circuit=append_measurement_basis_rotation(circuit,q,measurement_idx)
    circuit=fold_circuit(circuit,num_folding=num_folding)
    if backend_name=="qasm_simulator":
        circuit.measure(q,c)
    
    return circuit


def main():
     ## Testing
    N_states=4
    encoding="Graycode"
    N_qubits = int(np.ceil(np.log2(N_states)))

    measurement_idx="X"*N_qubits
    thetas = np.random.uniform(low=-np.pi/2, high=np.pi/2,size=N_states-1)

    circuit=variational_circuit(encoding,thetas,measurement_idx,"qasm_simulator",num_cnot_pairs=1,num_folding=1)
    circuit.draw(output="mpl",filename="circuit.pdf")

if(__name__ == "__main__"):
    main()


# def sparse_variational_circuit(thetas, measured_idx, backend_name, num_cnot_pairs=0, num_folding=0):
#     """ Creates a variational ansatz for the Jordan-Wigner encoding.

#     These circuits were defined in arXiv:1904.04338, and produce an ansatz state
#     over the occupation subset of the computational basis (|1000>, |0100>, etc.)
#     with generalized spherical coordinates as their amplitudes.

#     The version here is the same circuit, but with the qubit order inverted.
#     This was found to lead to greater stability in the SPSA optimization procedure
#     due to the relationships between of the variational parameters and the
#     "strength" of the basis states in the Hamiltonian.

#     Parameters:
#         thetas (np.array) : Angles parameterizing ansatz wavefunction.  Number of
#             angles is one fewer than the number of qubits in the circuit.

#         measured_idx (int) : Either 0, 1, or 2 corresponding to X, Y, and Z bases. The Pauli operators
#             in these Hamiltonians are either all X, all Y, or all Z, and never a mix, so the measurement
#             basis is easily specified and applied to all qubits.

#         backend_name (str) : name of the backend

#     Returns:
#         circuit (QuantumCircuit) : quantum circuit

#     """
#     # The number of parameters tells us the number of states
#     N_states = len(thetas) + 1
#     N_qubits = N_states
#     operations = []

#     if measured_idx > 3 or measured_idx < 0:
#         raise ValueError("Sparse Hamiltonians have only 3 sets of commuting operators.")

#     q, c = QuantumRegister(N_qubits), ClassicalRegister(N_qubits)
#     circuit = QuantumCircuit(q, c)

#     circuit.x(q[N_qubits-1])
#     operations.append({"x": (q[N_qubits-1])})
#     circuit.ry(thetas[0], q[N_qubits-2])
#     operations.append({"ry": (thetas[0], q[N_qubits-2])})
#     # Insert CNOT gates
#     for n_pairs in range(num_cnot_pairs):
#         circuit.cx(q[N_qubits-2], q[N_qubits-1])
#         operations.append({"cx": (q[N_qubits-2], q[N_qubits-1])})
#         circuit.cx(q[N_qubits-2], q[N_qubits-1])
#         operations.append({"cx": (q[N_qubits-2], q[N_qubits-1])})
#     circuit.cx(q[N_qubits-2], q[N_qubits-1])
#     operations.append({"cx": (q[N_qubits-2], q[N_qubits-1])})

#     # Recursive cascade
#     for control_idx in range(N_qubits - 2, 0, -1):
#         target_idx = control_idx - 1

#         #circuit.cry(thetas[control_idx], q[control_idx], q[target_idx])
#         circuit.ry(thetas[control_idx]/2, q[target_idx])
#         operations.append({"ry": (thetas[control_idx]/2, q[target_idx]) })
#         for n_pairs in range(num_cnot_pairs):
#             circuit.cx(q[control_idx], q[target_idx])
#             operations.append({"cx": (q[control_idx], q[target_idx])})
#             circuit.cx(q[control_idx], q[target_idx])
#             operations.append({"cx": (q[control_idx], q[target_idx])})
#         circuit.cx(q[control_idx], q[target_idx])
#         operations.append({"cx": (q[control_idx], q[target_idx])})
#         circuit.ry(-thetas[control_idx]/2, q[target_idx])
#         operations.append({"ry": (-thetas[control_idx]/2, q[target_idx]) })
#         for n_pairs in range(num_cnot_pairs):
#             circuit.cx(q[control_idx], q[target_idx])
#             operations.append({"cx": (q[control_idx], q[target_idx]) })
#             circuit.cx(q[control_idx], q[target_idx])
#             operations.append({"cx": (q[control_idx], q[target_idx]) })
#         circuit.cx(q[control_idx], q[target_idx])
#         operations.append({"cx": (q[control_idx], q[target_idx]) })
#         for n_pairs in range(num_cnot_pairs):
#             circuit.cx(q[target_idx], q[control_idx])
#             operations.append({"cx": (q[target_idx], q[control_idx]) })
#             circuit.cx(q[target_idx], q[control_idx])
#             operations.append({"cx": (q[target_idx], q[control_idx]) })
#         circuit.cx(q[target_idx], q[control_idx])
#         operations.append({"cx": (q[target_idx], q[control_idx]) })

#     if backend_name == 'qasm_simulator':
#         # Set the measurement basis - 0 is X, 1 is Y, 2 is Z
#         if measured_idx < 2:
#             if measured_idx == 1: # Y measurements
#                 for qubit_idx in range(N_qubits):
#                     circuit.sdg(q[qubit_idx])
#                     operations.append({"sdg": q[qubit_idx] })

#             for qubit_idx in range(N_qubits): # Y or X get Hadamards
#                 circuit.h(q[qubit_idx])
#                 operations.append({"h": q[qubit_idx] })

#     if( num_folding != 0 ):
#         for i_folding in range(num_folding):
#             for op in reversed(operations):
#                 gate = list(op.keys())[0]
#                 if( gate == "cx" ): circuit.cx( op[gate][0], op[gate][1] )
#                 if( gate == "ry" ): circuit.ry(-op[gate][0], op[gate][1] )
#                 if( gate == "x" ): circuit.x( op[gate] )
#                 if( backend_name == 'qasm_simulator' and gate == "h" ): circuit.h( op[gate] )
#                 if( backend_name == 'qasm_simulator' and gate == "sdg" ): circuit.s( op[gate] )
#             for op in operations:
#                 gate = list(op.keys())[0]
#                 if( gate == "cx" ): circuit.cx( op[gate][0], op[gate][1] )
#                 if( gate == "ry" ): circuit.ry( op[gate][0], op[gate][1] )
#                 if( gate == "x" ): circuit.x( op[gate] )
#                 if( backend_name == 'qasm_simulator' and gate == "h" ): circuit.h( op[gate] )
#                 if( backend_name == 'qasm_simulator' and gate == "sdg" ): circuit.sdg( op[gate] )

#     if( backend_name == 'qasm_simulator' ):
#         circuit.measure(q, c)

#     return circuit


# def dense_variational_circuit(thetas, measured_idx, backend_name, num_cnot_pairs=0, num_folding=0):
#     """ Creates a variational ansatz for the Gray code encoding.

#     These circuits have a more familiar 'variational form'. Since in the dense case we
#     are using all elements in the space, we can use something a bit more general. These
#     circuits are layers of Y rotations followed by entangling gates on all the qubits.

#     Parameters:

#         thetas (np.array) : Angles parameterizing ansatz wavefunction.  Number of
#             angles is one fewer than the number of qubits in the circuit.

#         measured_idx (int): The Paulis in this Hamiltonian are partitioned into N + 1 commuting
#             sets. For each set, only a single qubit will be measured in the X basis, and this is
#             what is passed. The case of Z is handled by measured_idx = N_qubits.

#         backend_name (str) : name of the backend

#     Returns:
#         circuit (QuantumCircuit) : quantum circuit

#     """
#     N_states = len(thetas) + 1
#     N_qubits = int(np.ceil(np.log2(N_states)))
#     operations = []

#     if measured_idx:
#         if measured_idx < 0 or measured_idx >= N_qubits:
#             raise ValueError("Invalid measurement index. Dense Hamiltonians have N_qubits + 1 commuting operators.")

#     q, c = QuantumRegister(N_qubits), ClassicalRegister(N_qubits)

#     circuit = QuantumCircuit(q, c)

#     num_remaining_thetas = len(thetas)
#     qubit_idx = 0
#     last_layer = False

#     while num_remaining_thetas > 0:
#         # If it's the last layer, only do as many CNOTs as we need to
#         if last_layer:
#             for i in range(num_remaining_thetas):
#                 for n_pairs in range(num_cnot_pairs):
#                     circuit.cx(q[i], q[(i+1)%N_qubits])
#                     operations.append({"cx":(q[i], q[(i+1)%N_qubits])})
#                     circuit.cx(q[i], q[(i+1)%N_qubits])
#                     operations.append({"cx":(q[i], q[(i+1)%N_qubits])})
#                 circuit.cx(q[i], q[(i+1)%N_qubits])
#                 operations.append({"cx":(q[i], q[(i+1)%N_qubits])})
#             for qubit in range(1, num_remaining_thetas+1):
#                 circuit.ry(thetas[num_remaining_thetas - 1], q[qubit])
#                 operations.append({"ry":(thetas[num_remaining_thetas - 1], q[qubit])})
#                 num_remaining_thetas -= 1
#             break
#         # If it's not the last layer, apply rotations and then increment counters
#         else:
#             circuit.ry(thetas[num_remaining_thetas - 1], q[qubit_idx])
#             operations.append({"ry":(thetas[num_remaining_thetas - 1], q[qubit_idx])})
#             qubit_idx += 1
#             num_remaining_thetas -= 1

#             if N_qubits >= 2:
#                 # If we have a full layer of parameters ahead, do the full CNOT cycle
#                 if qubit_idx == N_qubits and num_remaining_thetas > N_qubits:
#                     for i in range(N_qubits):
#                         for n_pairs in range(num_cnot_pairs):
#                             circuit.cx(q[i], q[(i+1)%N_qubits])
#                             operations.append({"cx":(q[i], q[(i+1)%N_qubits])})
#                             circuit.cx(q[i], q[(i+1)%N_qubits])
#                             operations.append({"cx":(q[i], q[(i+1)%N_qubits])})
#                         circuit.cx(q[i], q[(i+1)%N_qubits])
#                         operations.append({"cx":(q[i], q[(i+1)%N_qubits])})
#                     qubit_idx = 0
#                 # Otherwise, we are entering the last layer
#                 elif qubit_idx == N_qubits and num_remaining_thetas <= N_qubits:
#                     last_layer = True

#     if backend_name == 'qasm_simulator':
#         # Rotate the basis of the qubit that is being measured
#         if measured_idx is not None:
#             circuit.h(q[measured_idx])
#             operations.append({"h":q[measured_idx]})
#     if( num_folding != 0 ):
#         for i_folding in range(num_folding):
#             for op in reversed(operations):
#                 gate = list(op.keys())[0]
#                 if( gate == "cx" ): circuit.cx( op[gate][0], op[gate][1] )
#                 if( gate == "ry" ): circuit.ry(-op[gate][0], op[gate][1] )
#                 if( backend_name == 'qasm_simulator' and gate == "h" ): circuit.h( op[gate] )
#             for op in operations:
#                 gate = list(op.keys())[0]
#                 if( gate == "cx" ): circuit.cx( op[gate][0], op[gate][1] )
#                 if( gate == "ry" ): circuit.ry( op[gate][0], op[gate][1] )
#                 if( backend_name == 'qasm_simulator' and gate == "h" ): circuit.h( op[gate] )

#     if (backend_name == 'qasm_simulator'):
#         circuit.measure(q, c)

#     return circuit

