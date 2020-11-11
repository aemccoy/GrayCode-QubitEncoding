import numpy as np 
from scipy.linalg import lstsq
from pprint import pprint
from tqdm import tqdm

from scipy.linalg import eigh
from scipy.linalg import eig

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute, Aer
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.initial_states import Custom

from hamiltonian import *
from utils import *
from qiskit_circuits import *
import itertools

def normalization_coefficient(ncoefs,Ccoefs,r):
    """
    Compute the rth normalization coefficient via recurrence
    """
    nr=ncoefs[r-1]
    Cr=Ccoefs[r-1]
    return nr/np.sqrt(Cr)

def normalization_coefficients(Ccoefs):
    """
    Compute normalization coefficients from overlaps 
    """
    ncoefs=np.zeros(len(Ccoefs))
    ncoefs[0]=1/np.sqrt(Ccoefs[0])
    for r in range(1,len(Ccoefs)):
        ncoefs[r]=normalization_coefficient(ncoefs,Ccoefs,r)
    
    return ncoefs


def Krylov_indices(ncoefs,krylov_threshold=.99999,starting_index=0):
    krylov_indices=[starting_index]
    for i in range(starting_index+2,len(ncoefs),2):
        j=krylov_indices[-1]
        k=int((i+j)/2)
        regularization_factor=(ncoefs[i]*ncoefs[j])/(ncoefs[k]**2)
        if regularization_factor<krylov_threshold:
            krylov_indices.append(i)
    return krylov_indices

def Krylov_matrices(krylov_indices,ncoefs,energies,threshold=1e-2):
    dim=len(krylov_indices)
    
    ## Construct overlap matrix
    T=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            l=krylov_indices[j]
            lp=krylov_indices[i]
            k=int((l+lp)/2)
            T[i,j]=(ncoefs[l]*ncoefs[lp])/(ncoefs[k]**2)

    ## Elimiate small eigenvalues form overlap matrix
    eigs,evecs=eig(T)
    D=np.zeros((dim,dim))
    for i in range(dim):
        if abs(eigs[i].real)>threshold:
            D[i,i]=eigs[i]
            
    T=np.dot(evecs.transpose(),np.dot(D,evecs))  
            
    ## Construct H matrix 
    H=np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            l=krylov_indices[j]
            lp=krylov_indices[i]
            k=int((l+lp)/2)
            H[i,j]=T[i,j]*energies[k]
            
    return T,H

def do_QLanczos(Ccoefs,energies,krylov_threshold=0.99999,starting_index=0):
    ## Do QLanczos on computed vectors 
    ncoefs=normalization_coefficients(Ccoefs)
    krylov_indices=Krylov_indices(ncoefs,krylov_threshold=krylov_threshold,starting_index=starting_index)
    print(f"Dim Krylov space: {len(krylov_indices)}")
    ## Construct H and overlap matrix T in Krylov basis
    T_krylov,H_krylov=Krylov_matrices(krylov_indices,ncoefs,energies,threshold=.5)
    
    ## Solve generalized eigenproblem
    eigs,vecs=eig(H_krylov,T_krylov)

    # ## Identify the return lowest eigenvalue
    # idx = eigs.argsort()[0]
    # return eigs[idx].real

    return eigs.real



def eliminate_outliers(energies):
    temp=[]
    for e in energies:
        if abs(e)<=1e2: 
            temp.append(e)
    energies=temp

    mean=np.mean(energies)
    sd=np.std(energies)
    survivors=[]
    for e in energies:
        if abs(e)<=(abs(mean)+2*sd):
            survivors.append(e)
        else:
            print(f"{e:.5f} killed")
    return survivors

def remove_outliers(energies):
    e_len=len(energies)
    diff=9999
    while diff>0:
        energies=eliminate_outliers(energies)
        diff=e_len-len(energies)
        e_len=len(energies)
    return energies

def analyze_qlanczos_results(results_filename):
        ## Each row corresponds to a given time step
    results=np.load(results_filename,allow_pickle=True)
    # print(results)
    temp=[]
    for result in results:
        temp.append(result[0])

    corrected_results=remove_outliers(temp)
    averages=np.mean(corrected_results)
    stdevs=np.std(corrected_results)
    return averages,stdevs
