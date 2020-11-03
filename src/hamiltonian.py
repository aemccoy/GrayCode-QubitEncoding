import numpy as np
from itertools import product, chain
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner 
from openfermion.transforms import get_sparse_operator 
from openfermion.utils import get_ground_state 
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator

from utils import * 

def generate_relative_states(Nmax,J,L0=-1):
    """
    Generate deuteron basis for J subject to antisymmetry constraints
    """
    ## For deuteron, J=1
    i=0
    labels={}

    Nlist=list(range(0,Nmax+1,2))
 
    for T in [0,1]:
        for N in Nlist:
            ## If restricted to specific L channel
            if L0==-1:
                L_values=list(range(N%2,N+1,2))
            else:
                L_values=[L0]
            for L in L_values:
                for S in [0,1]:
                
                    if (N+S+T)%2!=1:
                        continue
                    if abs(L-S)<=J and (L+S)>=J:
                        labels[(N,L,S,J,T)]=i
                        i += 1

    return labels



def get_interaction(filename,basis):
    """
    Read in interaction from file. 
    File format is given in spncci/basis/lsjt_operator.h
    """
#     basis=generate_relative_states(Nmax,J)
    dim=len(basis)
    V=np.zeros((dim,dim))
    # interaction={}
    f=open(filename,'r')
    # Skipping header lines
    for i in range(6):
        next(f)
    ## Get relevant header information 
    operator_info=f.readline()
    operator_basis_info=f.readline()
    #for each line, get labels and matrix element, look up 
    for line in f:
        labels = tuple(map(int,line.split()[:11]))
        value = float(line.split()[11])
        [T0, Np, Lp, Sp, Jp, Tp, N, L, S, J, T]=labels
        statep = (Np, Lp, Sp, Jp, Tp)
        state = (N, L, S, J, T)
        if (statep in basis) and (state in basis):
            i = basis[state]
            ip = basis[statep]
            ## Sums over T0 values if relevant 
            V[ip,i] += value
            if ip!=i:
                V[i,ip] += value

    f.close()
#     print(V)
    return V

def toy_interaction(basis):
    """
    Read in interaction from file. 
    File format is given in spncci/basis/lsjt_operator.h
    """
#     basis=generate_relative_states(Nmax,J)
    dim=len(basis)
    V=np.zeros((dim,dim))
    # Skipping header lines

    for statep in basis:
        [Np,Lp,Sp,Jp,Tp]=statep
        ip=basis[statep]
        for state in basis:
            [N,L,S,J,T]=state
            i=basis[state]
            n=(N-L)/2
            n_prime=(Np-Lp)/2
            if(n==n_prime) and (n==0) and ((L,S,J,T)==(Lp,Sp,Jp,Tp)):
                V[ip,i] = -5.68658111

    return V


def Tme(Np,Lp,Sp,Jp,Tp,N,L,S,J,T,hw,positive_origin=True):
    """
    Compute kinetic energy matrix element for given hw values
    based on formula given in prc-93-2016-044332-Binder
    """
    # Kinetic energy
    if (Sp!=S) or (Tp!=T) or (Jp!=J) or (Lp!=L):
        return 0
    
    if positive_origin:
        sigma=1
    else:
        sigma=-1

    n=int((N-L)/2);
    tme=0.0
    if Np==N:
        tme = hw/2*(2*n+L+1.5)
    if Np==(N-2):
        tme = sigma*hw/2*np.sqrt(n*(n+L+0.5))
    if Np==(N+2):
        tme = sigma*hw/2*np.sqrt((n+1)*(n+L+1.5))

    return tme

def get_kinetic_energy(basis,hw,positive_origin=True):
    """
    Construct kinetic energy matrix 

    Input 
    """
    
    dim=len(basis) 
    T_matrix=np.zeros((dim,dim))
    for statep in basis:
        [Np,Lp,Sp,Jp,Tp]=statep
        ip=basis[statep]
        for state in basis:
            [N,L,S,J,T]=state
            i=basis[state]
            T_matrix[ip,i]=Tme(Np,Lp,Sp,Jp,Tp,N,L,S,J,T,hw,positive_origin)
    return T_matrix


def hamiltonian_matrix(Nmax,hw,J,interaction_filename,positive_origin=True):
    """
    Get interaction from file and constructs Hamiltonian matrix.

    Input:
        Nmax(int) : Nmax of basis
        hw (float) : Harmonic oscillator basis parameter
        J (float or int) :  Angular momenum of basis
        interaction_filename (str) : file name of interaction or identifier 
                if "toy_hamiltonian", matrix constructed for toy deuteron problem
                    of [REF]

    Returns: 
        Hamiltonian matrix
    """
    if interaction_filename=="toy_hamiltonian":
        basis=generate_relative_states(Nmax,J,L0=0)
        V_matrix=toy_interaction(basis)
        hw=7
        positive_origin=False

    else:
        basis=generate_relative_states(Nmax,J)
        V_matrix=get_interaction(interaction_filename,basis)
    
    #Construct kinetic eneryg matrix 
    T_matrix=get_kinetic_energy(basis,hw,positive_origin)
    
    return T_matrix+V_matrix

def H_to_weighted_pauli(self):
    """
    Converts Hamiltonian operator into a WeightedPauliOperator

    Input: 
        H (hamiltonian.EncodingHamiltonian) : Qubit Hamiltonian

    Returns (WeightedPauliOperator) : Qubit Hamiltonian expressed as a WeightedPauliOperator

    """
    H_pairs=[(self.pauli_coeffs[k], Pauli.from_label(k)) for k in self.pauli_coeffs]
    return WeightedPauliOperator(H_pairs)   



class EncodingHamiltonian():

    # def __init__(self, N_qubits, N_states, qiskit_order=True):
    def __init__(self, H_matrix, qiskit_order=True):
        # self.N_qubits = N_qubits
        self.N_states = np.size(H_matrix,0)
        self.ferm_rep = self._generate_ferm_rep(H_matrix)
        self.qiskit_order = qiskit_order

    def _generate_ferm_rep(self,H_matrix):
        """ Construct the Fermionic representation of this Hamiltonian"""
        
        # Initialize 
        H = FermionOperator('1^ 1', 0)

        for n, n_prime in product(range(self.N_states), repeat=2):

            H += FermionOperator(f"{n_prime+1}^ {n+1}", H_matrix[n_prime,n])

        return H
 
    
class GrayCodeHamiltonian(EncodingHamiltonian):
    def __init__(self, H, qiskit_order=True):
        """ Class for Gray code encoding that uses N qubits to represent 2^N states.  [TODO:REF]

        Parameters:
            N_states (int) : The number of harmonic oscillator states to consider. For this
                encoding, the number of qubits will be Ceiling[log2[N_states]].

            qiskit_order (bool,optional) : Determines whether to order the qubits in qiskit order, i.e.
                in "reverse" as compared to the typical ordering. Default : True.

        """
        super(GrayCodeHamiltonian, self).__init__(H, qiskit_order)

        N_states=np.size(H,0)
        N_qubits = int(np.ceil(np.log2(N_states)))

        if N_states == 1:
            N_qubits = 1

        self.N_qubits=N_qubits

        # Get the order of the states in the gray code
        self.state_order = gray_code(self.N_qubits)
        
        # Map order of states onto qiskit basis??? Liv?
        self.permutation = [int("0b" + x, 2) for x in self.state_order] 

        # Get pauli representation for H acting on qubit states ordered by gray code
        self.pauli_rep = self._build_pauli_rep(H) 

        # self.to_dict=qubit_operator_to_dict(self)
        self.pauli_coeffs = qubit_operator_to_dict(self) 
        
        self.pauli_partitions = self._pauli_partitions()
        
        # H represented as weighted pauli operator
        self.weighted_pauli=H_to_weighted_pauli(self)
        
        self.n_partitions = len(self.pauli_partitions.keys())
        
        self.matrix = self._to_matrix()
    
    def _build_pauli_rep(self,H):
        """
        Get pauli representation for H acting on qubit states ordered by gray code
        """
        ## Generate number of states 
        N_states=np.size(H,0)
        
        ## Get number of qubits 
        if N_states==1:
            N_qubits=1
        else:
            N_qubits = int(np.ceil(np.log2(N_states)))
        
        ## Construct graycode for qubits 
        gc_states=gray_code(N_qubits)
        
        ## initialize H
        full_operator = QubitOperator()
        for ip in range(0,N_states,1):
            # for i in range(0,ip+1,1):
            for i in range(0,N_states,1):
                Hme=H[ip,i]
                if Hme == 0.0:
                    continue
                s1=gc_states[i]
                s2=gc_states[ip]
                term=graycode_operator(s1,s2)
                full_operator += Hme*term

        return full_operator 



    def _to_matrix(self):
        ## If in qiskit ordering, need to flip pauli strings to get back to left to right ordering for matrix rep
        if self.qiskit_order:
            return reduce(lambda x, y: x + y, [p[1] * get_pauli_matrix(p[0][::-1]) for p in self.pauli_coeffs.items()]).real
        else:
            return reduce(lambda x, y: x + y, [p[1] * get_pauli_matrix(p[0]) for p in self.pauli_coeffs.items()]).real


    def _pauli_partitions(self):
        """
        Partition pauli terms in Hamiltonian into commuting sets 

        Returns:
            (dictionary) : Dictionary keyed by 
        """
        pauli_dict=self.pauli_coeffs
        commuting_sets=get_commuting_sets(list(pauli_dict.keys()))

        return commuting_sets

    def _separate_coeffs(self):
        """ Pulls out the coefficients of each Pauli and stores in a dictionary separate.
        Useful for computing the expectation value because we can look up coeffs easily.
        """
        all_paulis = {}
        for set_idx, measurement_setting in self.pauli_partitions.items():
            for pauli, coeff in measurement_setting.items():
                all_paulis[pauli] = coeff.real
        return all_paulis

class JordanWignerHamiltonian(EncodingHamiltonian):
    def __init__(self, H, qiskit_order=True):    
        """ Class for the Jordan-Wigner encoded Hamiltonian. 

        Parameters: 
            H (np.matrix) : Hamiltonian matrix to be encoded 
            
            qiskit_order (bool): Determines whether to order the qubits in qiskit order, i.e.
                in "reverse" as compared to the typical ordering. 
        """
        super(JordanWignerHamiltonian, self).__init__(H, qiskit_order)

        self.N_qubits=self.N_states
        self.pauli_rep = self._build_pauli_rep()
        self.pauli_coeffs = qubit_operator_to_dict(self)
        self.pauli_partitions = self._pauli_partitions()
        # H represented as weighted pauli operator
        self.weighted_pauli=H_to_weighted_pauli(self)
        self.n_partitions = len(self.pauli_partitions.keys())
        
        self.matrix = self._to_matrix() # Numerical matrix

    def _build_pauli_rep(self):
        pauli_rep=jordan_wigner(self.ferm_rep).terms.items()
        # The Jordan-Wigner transform from OpenFermion gives us Paulis where 
        # the qubits are 1-indexed. Convert to 0 indexing, and if we are using
        # Qiskit order, we also have to reverse the order of the Paulis.
        new_pauli_rep = QubitOperator()

        for pauli, coeff in pauli_rep:
            operator_string = ""
            for qubit in pauli:
                operator_string += (qubit[1] + str(qubit[0] - 1) + " ")
            new_pauli_rep += coeff.real * QubitOperator(operator_string)

        return new_pauli_rep

    
    def _to_matrix(self):
        mat = np.zeros((self.N_states, self.N_states))
        ## Returns matrix representation of qubit operator
        ## If in qiskit ordering, need to flip pauli strings to get back to left to right ordering for matrix rep
        if self.qiskit_order:
            return reduce(lambda x, y: x + y, [p[1] * get_pauli_matrix(p[0][::-1]) for p in self.pauli_coeffs.items()]).real
        else:
            return reduce(lambda x, y: x + y, [p[1] * get_pauli_matrix(p[0]) for p in self.pauli_coeffs.items()]).real

        ## Old version, returns NstatexNstate matrix 
        # # For each term in the Hamiltonian, populate the relevant entry in the matrix
        # for ferm_op in self.ferm_rep:
        #     dag_op, op = list(ferm_op.terms.keys())[0]
        #     dag_idx, op_idx = dag_op[0]-1, op[0]-1
        #     mat[dag_idx, op_idx] = ferm_op.terms[(dag_op, op)]

        # return mat

    def _pauli_partitions(self):
        """
        Partition pauli terms in Hamiltonian into commuting sets 
        """
        pauli_dict=self.pauli_coeffs
        commuting_sets=get_commuting_sets(list(pauli_dict.keys()))

        return commuting_sets