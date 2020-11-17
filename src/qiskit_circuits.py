from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

import numpy as np
import sys

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


def initialize_circuit(q,c,initial_state="single_state",encoding="gray_code"):
    """
    Initialize circuit.  
    
    input:
        q (qiskit.circuit.quantumregister.QuantumRegister) : qubits
        initial_state(str or np.array,optional) : initial state of circuit

            "single_state" (str,default) : starting state is qubit state with all qubits in zero state
                    If encoding="gray_code", initialize to |00>
                    If encoding="jordan_wigner", Initalize to |01>

            "zeros"(str,obsolete) : same as single state for Graycode

            "uniform" (str) : Uniform superposition of all qubit states 
                Currently only implement for Graycode

             array(np.array) : normalized array of len(2**n_qubits)

        encoding(str,optional) : Encoding used. Either "Graycode or JordanWigner"
    
    returns:
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : initialized quantum circuit
    
    """    
    if encoding!="gray_code" and encoding!="jordan_wigner":
        raise ValueError("Invalid encoding for circuit.")

    ## Sets parameters to only create circuit but not add initial state
    if initial_state == "zeros" or initial_state == None:
        initial_state = "single_state"
        encoding = "gray_code"
        
    if isinstance(initial_state, str):
        if (initial_state=="single_state") and (encoding == "gray_code"):
            circuit = QuantumCircuit(q,c)

        elif initial_state=="single_state" and encoding == "jordan_wigner":
            circuit = QuantumCircuit(q,c)
            circuit.x(q[0])

        elif initial_state=="uniform" and encoding == "gray_code":
            circuit = QuantumCircuit(q,c)
            circuit.h(q)
        else:
            sys.exit(f"{initial_state} for {encoding} is not currently a valid option")
    else:
        circuit = QuantumCircuit(q,c)
        circuit.initialize(initial_state,q)
    return circuit


def fold_circuit(circuit,number_circuit_folds):
    """
    Applies and then inverts circuit number_circuit_folds times before finally running the circuit
    Used for error extrapolation? 
    
    Input:
        circuit (QuantumCircuit) : quantum circuit 
        
        number_circuit_folds (int) : Number of times circuit is "folded"
        
    Returns:
        circuit (QuantumCircuit) : folded circuit
    
    """
    folded_circuit=circuit
    for n in range(number_circuit_folds):
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


def variational_circuit(encoding,thetas,measurement_idx,backend_name,num_cnot_pairs=0,number_circuit_folds=0):
    """
    Construct variational circuit 

    input:
        encoding (string) : string identifier for which type of variational circuit to run 
                "gray_code" : variational ansatz for dense Gray code encoding
                "jordan_wigner" : variational ansatz for Jordan Wigner encoding

    returns:
        circuit (QuantumCircuit) : variational circuit
    """
    N_states=len(thetas)+1
    if encoding=="gray_code":
        N_qubits = int(np.ceil(np.log2(N_states)))
    elif encoding == "jordan_wigner":
        N_qubits=N_states
    else:
        raise ValueError("Invalid encoding for variational circuit.")
        
    q, c = QuantumRegister(N_qubits), ClassicalRegister(N_qubits)
    circuit = QuantumCircuit(q, c)
    
    if encoding=="gray_code":
        circuit=append_Gray_code_variational_ansatz(circuit,q,thetas,num_cnot_pairs)
    else: # must be "jordan_wigner" else exception raised earlier 
        circuit=append_Jordan_Wigner_variational_ansatz(circuit,q,thetas,num_cnot_pairs)
        
    if backend_name=="qasm_simulator":
#         print(f"m idx: {measurement_idx}")
        circuit=append_measurement_basis_rotation(circuit,q,measurement_idx)
    circuit=fold_circuit(circuit,number_circuit_folds=number_circuit_folds)
    if backend_name=="qasm_simulator":
        circuit.measure(q,c)
    
    return circuit


def get_operator_list_from_qasm_string(qasm_string,classical_registers=False):
    """
    Get list of qubit operators from qasm string
    
    input:
        qasm_string(str) : String describing circuit generated by circuit.qasm()
            string starts with 
                OPENQASM 2.0;\n
                include "qelib1.inc";\n
                qreg qid[numq];\n
                creg cid[numc];\n
                <gates>
                <measurements>
            
    """
    ## Break up string by line breaks
    circuit_strings=qasm_string.split(";\n")

    ## Get quantum registers id and num q registers 
    qreg,qid=circuit_strings[2].split(" ")
    assert(qreg == 'qreg')   
    qid,numq = qid[:-1].split("[")
    numq=int(numq)
    cid=""
    if classical_registers:
        ## Get classical registers id and num c registers 
        creg,cid=circuit_strings[3].split(" ")
        cid,numc = cid[:-1].split("[")
        numc=int(numc)
        gate_starting_index=4
    else:
        gate_starting_index=3
    ## Iterate through remaining operators and if not a measurement,
    ## Reformat operator string and break up into terms 
    operators=[]
    pi_conversions={'pi/2':np.pi/2,'-pi/2':-np.pi/2,'pi':np.pi, '-pi':-np.pi}
    for op in circuit_strings[gate_starting_index:]:
    #     print("\n"+op)
        op=op.replace("u2(0,pi)","h")
        for s in [qid,cid,"[","]"]:
            op=op.replace(s,"")
        for s in ["(",")","->",","]:
            op=op.replace(s," ")
        op=op.split()
        if len(op)>0 and op[0]!='measure':
            op=[pi_conversions[e] if e in pi_conversions else e for e in op]
            operators.append(op)
    return operators


def append_circuit_from_qasm_list(circuit,q,operators,num_cnot_pairs=0):
    """
    Construct circuit from list of operators allowing for 
    cnot pairs to be inserted
    """    
    for op in operators:
        if op[0] == "z":
            circuit.z(int(op[1]))
        elif op[0] == "x":
            circuit.x(int(op[1]))
        elif op[0] == "y":
            circuit.y(int(op[1]))
        elif op[0] == "h":
            circuit.h(int(op[1]))
        elif op[0] == "s":
            circuit.s(int(op[1]))
        elif op[0] == "cx":
            add_cnot_gate(circuit,q,int(op[1]),int(op[2]),num_cnot_pairs)
        elif op[0] == "rx":
            circuit.rx(float(op[1]),int(op[2]))
        elif op[0] == "ry":
            circuit.ry(float(op[1]),int(op[2]))
        elif op[0] == "u1":
            circuit.u1(float(op[1]),int(op[2]))
        elif op[0] == "u2":
            circuit.u2(float(op[1]),float(op[2]),int(op[3]))
        elif op[0] == "u3":
            circuit.u3(float(op[1]),float(op[2]),float(op[3]),int(op[4]))
        elif op[0] == "r":
            if op[2] == "pi/2":
                circuit.ry(float(op[1]),int(op[3]))
            elif op[2] == "0":
                circuit.rx(float(op[1]),int(op[3]))
            else:
                circuit.r(float(op[1]),float(op[2]),int(op[3]))
        
        else:
            raise ValueError(f"Gate {op} not implemented.")
    return circuit


def append_evolution_circuit(q,evolution_set,time,circuit,num_cnot_pairs=0):
    """
    Append evolution exp(-iAt) onto circuit for each time step t
    
    input:
        q (qiskit.circuit.quantumregister.QuantumRegister) : qubits
        evolution_set () : List of Weighted Pauli operators by which the circuit is evolved at each timestep
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : quantum circuit 

    return
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit) : updated quantum circuit
    """
    evolution_circuit=QuantumCircuit(q)
    for A in evolution_set: 
        #Append next step to circuit for each A
        evolution_circuit += A.evolve(
            None, evo_time=time, num_time_slices=1,
            quantum_registers=q
            )
    if num_cnot_pairs==0:
        circuit=circuit+evolution_circuit
    else:
        qasm_string=evolution_circuit.decompose().qasm()
        operators=get_operator_list_from_qasm_string(qasm_string,classical_registers=False)
        circuit=append_circuit_from_qasm_list(circuit,q,operators,num_cnot_pairs)

    return circuit  


def main():
     ## Testing
    N_states=4
    encoding="gray_code"
    N_qubits = int(np.ceil(np.log2(N_states)))

    measurement_idx="X"*N_qubits
    thetas = np.random.uniform(low=-np.pi/2, high=np.pi/2,size=N_states-1)

    circuit=variational_circuit(encoding,thetas,measurement_idx,"qasm_simulator",num_cnot_pairs=1,number_circuit_folds=1)
    circuit.draw(output="mpl",filename="circuit.pdf")


if(__name__ == "__main__"):
    main()



