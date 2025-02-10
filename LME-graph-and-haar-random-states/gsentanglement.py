import numpy as np
from typing import List, Tuple, Callable
import itertools
import math
import pyswarms as ps

########################################################################################################################################################################################

"""
Initializing some regularly used quantities and quantum objects such as |0>, |1>, |0><0|, |+>, |+><+|, X, Y, pi, e...
"""

e = np.e
pi = np.pi

plusState = np.matrix([[1/2**.5], [1/2**.5]])
minusState = np.matrix([[1/2**.5], [-1/2**.5]])
zeroState = np.matrix([[1],[0]])
oneState = np.matrix([[0],[1]])
zeroStateDensity = zeroState*zeroState.T
oneStateDensity = oneState*oneState.T
plusStateDensity = plusState*plusState.T
minusStateDensity = minusState*minusState.T

X = np.matrix([[0, 1], [1, 0]])
Y = np.matrix([[0, -1j], [1j, 0]])
Z = np.matrix([[1, 0], [0, -1]])
I = np.matrix([[1, 0], [0, 1]])
H = 2**(-1/2)*np.matrix([[1,1],[1,-1]])
iXSqrt = np.exp(1j*pi/4)*plusStateDensity + np.exp(-1j*pi/4)*minusStateDensity
iZSqrt = np.exp(1j*pi/4)*zeroStateDensity + np.exp(-1j*pi/4)*oneStateDensity

########################################################################################################################################################################################

def partialTrace(traceOutSystems: List[float], densityMatrix: np.matrix) -> np.matrix:

    """
    Calculates the partial trace of a qubit state given its corresponding density matrix and the indexes of the traced out systems.
    
    Parameters:
    tracedOutSystems (List[float]): The list of the traced out systems.
    densityMatrix (np.matrix): The density matrix of the system.

    Returns:
    np.matrix: The partial trace.
    """

    matrixSum = 0

    totalSystems = int(np.log2(len(densityMatrix)))
    
    binaryVectors = list(map(list, itertools.product([0, 1], repeat=len(traceOutSystems))))

    for i in range(len(binaryVectors)): 
        prod = 1
        for j in range(1, totalSystems+1):
            if traceOutSystems.count(j)>0: #if j is in the list traceOutSystems
                if binaryVectors[i][traceOutSystems.index(j)]==0: 
                    prod = np.kron(prod, zeroState)
                else:
                    prod = np.kron(prod, oneState)
            else:
                prod = np.kron(prod, I)
        matrixSum += np.matmul(np.matmul(prod.getH(), densityMatrix), prod)

    return matrixSum

########################################################################################################################################################################################

def S(phi: float) -> np.matrix:

    """
    Creates the S phase gate given an angle phi.
    
    Parameters:
    phi (float): The angle of the phase.

    Returns:
    np.matrix: The S phase gate.
    """
    
    return np.matrix([[1, 0], [0, e**(1j*phi)]])

def YProd(N):

    """
    Returns a tensor product of N Pauli Y matrices.

    Parameters:
    N (int): The number of Y matrices to be multiplied.

    Returns:
    np.matrix: The tensor product of Y matrices.
    """

    YProd = 1

    for i in range(N):
        YProd = np.kron(YProd, Y)

    return YProd

def U(params: np.array) -> np.matrix:

    """
    Creates the single-qubit U operator using the Nielsen and Chuang parametrization of a 2x2 unitary matrix.
    
    Parameters:
    params (np.array): An array with the four parameters.

    Returns:
    np.matrix: The parametrized U gate.
    """
    
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    delta = params[3]
    #Nielsen and Chuang parameterization of 2x2 unitary matrix
    return np.matrix([[np.exp(1j*(alpha - beta/2 - delta/2))*np.cos(gamma/2),
                       -np.exp(1j*(alpha - beta/2 + delta/2))*np.sin(gamma/2)],
                      [np.exp(1j*(alpha + beta/2 - delta/2))*np.sin(gamma/2),
                       np.exp(1j*(alpha + beta/2 + delta/2))*np.cos(gamma/2)]])

def U3(params: np.array) -> np.matrix:

    """
    Creates the single-qubit U operator using the U3 parametrization like the one used in qiskit and pennylane.
    
    Parameters:
    params (np.array): An array with the four parameters.

    Returns:
    np.matrix: The parametrized 3 gate.
    """
    
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    #Nielsen and Chuang parameterization of 2x2 unitary matrix
    return np.matrix([[np.cos(alpha/2),
                       -np.exp(1j*gamma)*np.sin(alpha/2)],
                      [np.exp(1j*beta)*np.sin(alpha/2),
                       np.exp(1j*(beta+gamma))*np.cos(alpha/2)]])

def Meas(params: np.array) -> np.matrix:

    """
    Creates the single-qubit U operator using the U3 parametrization like the one used in qiskit and pennylane.
    
    Parameters:
    params (np.array): An array with the four parameters.

    Returns:
    np.matrix: The parametrized 3 gate.
    """
    
    alpha = params[0]
    beta = params[1]
    #Nielsen and Chuang parameterization of 2x2 unitary matrix
    return np.matrix([[np.cos(alpha/2),
                       -np.exp(-1j*beta)*np.sin(alpha/2)],
                      [np.exp(1j*beta)*np.sin(alpha/2),
                       np.cos(alpha/2)]])

########################################################################################################################################################################################

def CU(control: float, target: float, nqubits: float, U: np.matrix) -> np.matrix:

    """
    Creates the CPhase operator given a control qubit, a target qubit, the overall number of qubits, and the phase between them.

    Parameters:
    control (float): The control qubit index.
    target (float): The target qubit index.
    nqubits (float): The overall number of qubits.
    phi (float): The phase operator between the qubits.

    Returns:
    np.matrix: The controlled-phase gate.
    """

    cp0 = 1
    cp1 = 1

    for i in range(nqubits):
        if (i==control):
            cp0 = np.kron(cp0, zeroStateDensity)
            cp1 = np.kron(cp1, oneStateDensity)
        elif (i==target):
            cp0 = np.kron(cp0, I)
            cp1 = np.kron(cp1, U)
        else:
            cp0 = np.kron(cp0, I)
            cp1 = np.kron(cp1, I)

    return np.matrix(cp0+cp1)

########################################################################################################################################################################################

def generateGraphState(adjacencyMatrix: np.array, phi: float) -> np.matrix:

    """
    Creates a graph state given the adjacency matrix of a graph.

    Parameters:
    adjacencyMatrix (np.array): The adjacency matrix of the graph.
    phi (float): The phase between the qubits.

    Returns:
    np.matrix: The graph state.
    """
    
    nqubits = len(adjacencyMatrix)
    
    state = 1

    for i in range(nqubits):
        state = np.kron(state, plusState)
        
    for i in range(nqubits-1):
        for j in range(i+1, nqubits):
            if (adjacencyMatrix[i][j]==1):
                state = CU(i, j, nqubits, S(phi))*state

    return state

########################################################################################################################################################################################

def generatePostMeasurementEnsamble(measuredOutSystems: List[float], initialState: np.matrix, measurements: List[np.matrix]) -> Tuple[float, np.matrix]:

    """
    This function will generate the post measurement ensamble given a list with the measured out systems and their associated measurement observable.

    Parameters:
    measuredOutSystems (List[float]): List with the systems being measured.
    initialState (np.matrix): The state before the measurement.
    measurements (List[np.matrix]): The observables being measured.

    Returns:
    List[float, np.matrix]: The ensamble
    """

    nMeasurements = len(measuredOutSystems)

    nSystems = int(np.log2(len(initialState)))

    combinations = list(itertools.product([0, 1], repeat=nMeasurements))

    nCombinations = 2**nMeasurements

    ensamble = []

    for i in range(nCombinations):
        projector = 1
        iter = 0
        for j in range(1, nSystems+1):
            if (j in measuredOutSystems):
                if (combinations[i][iter]==0):
                    projector = np.kron(projector, zeroState.getH()@measurements[iter].getH())
                    iter = iter + 1
                else:
                    projector = np.kron(projector, oneState.getH()@measurements[iter].getH())
                    iter = iter + 1
            else:
                projector = np.kron(projector, I)        
        projectedVec = projector@initialState
        probability = abs(projectedVec.getH()@projectedVec)
        measuredOutState = projectedVec/np.sqrt(probability)
        ensamble.append([probability.item(0,0), measuredOutState])

    return ensamble

########################################################################################################################################################################################

def nTangle(densityMatrix: np.matrix) -> float:

    """
    This function will calculate the n-tangle of a state.

    Parameters:
    densityMatrix (np.matrix): The density matrix of the system.

    Returns:
    float: The n-tangle of the state.
    """

    totalSystems = int(np.log2(len(densityMatrix)))

    yProd = 1

    for i in range(totalSystems):
        yProd = np.kron(yProd, Y)

    return np.sqrt(abs((densityMatrix@yProd@densityMatrix.conjugate()@yProd).trace().item(0,0)))

########################################################################################################################################################################################

def gmeConcurrence(densityMatrix: np.matrix) -> float:

    """
    This function calculates the Genuine Multipartite Entanglement of a given density matrix.

    Parameters:
    densityMatrix (np.matrix): The density matrix of the system.

    Returns:
    float: The Genuine Multipartite Entanglement.
    """

    totalSystems = int(np.log2(len(densityMatrix)))
    
    binaryVectors = [list(vector) for vector in itertools.product([0, 1], repeat=totalSystems) if vector != tuple([0]*totalSystems) and vector != tuple([1]*totalSystems)]

    totalCombinations = len(binaryVectors)
    
    minConcur = 1
    
    for vector in range(totalCombinations):
        
        traceOutSystems = []
        
        for digit in range(totalSystems):
            if binaryVectors[vector][digit] == 1:
                traceOutSystems.append(digit+1)

        marginal = partialTrace(traceOutSystems, densityMatrix)

        concurrence = np.sqrt(2*(1 - np.trace(marginal@marginal)))

        if concurrence <= minConcur:
            minConcur = concurrence

    return abs(minConcur)

########################################################################################################################################################################################

def concentratableEntanglement(densityMatrix: np.matrix) -> float: #, s: List[float]

    """
    This function calculates the concentratable entanglement of a state given a chosen subset of qubits.

    Parameters:
    densityMatrix (np.matrix): The density matrix of the system.
    s (List[float]): The chosen subset.

    Returns:
    float: The concentratable entanglement.
    """

    s = [1, 2, 3, 4, 5]

    subSetCardinality = len(s)

    totalSystems = int(np.log2(len(densityMatrix)))

    power_set = []

    for i in range(1, 2**subSetCardinality):  # Start from 1 to exclude the empty set

        subset = [s[j] for j in range(subSetCardinality) if (i >> j) & 1]

        power_set.append(subset)

    ce = 1-1/(2**subSetCardinality)

    powerSetCardinality = len(power_set)

    for i in range(powerSetCardinality):

        d = partialTrace(power_set[i], densityMatrix)

        ce = ce - 1/2**(subSetCardinality)*np.trace(d*d)

    return ce

########################################################################################################################################################################################

def opSqrt(rho):

    """ 
    This function returns the operator square root of a density matrix.

    Parameters:
    rho (np.matrix): The density matrix of the system.

    Returns:
    np.matrix: The operator square root.
    """

    nEvals = (np.linalg.eigh(rho)[0]).size
    opSqrtSum = 0

    for i in range(nEvals):
        lambdai = np.linalg.eigh(rho)[0][i]
        ei = (np.linalg.eigh(rho)[1])[:,i]
        opSqrtSum += np.sqrt(abs(lambdai))*ei@ei.getH()

    return opSqrtSum

def F(rho1, rho2):

    """
    This function returns the fidelity of two density matrices.

    Parameters:
    rho1 (np.matrix): The first density matrix.
    rho2 (np.matrix): The second density matrix.

    Returns:
    float: The fidelity.
    """

    return abs(np.trace(opSqrt(opSqrt(rho2)@rho1@opSqrt(rho2))))

########################################################################################################################################################################################

def LME(measParamsMatrix: List[np.array], initialState: np.matrix, measureOutSystems: List[float], entanglementMeasure: Callable[[np.matrix], float]) -> List[float]:

    """
    This function calculates the Localizable Multipartite Entanglement of a given entanglement measure given the a list of parameters for single-qubit measurements, an initial state,
    the measured out systems and the entanglement measure.

    Parameters:
    measParamsMatrix (np.array): The matrix with the parameters of the single qubit measaurements.
    initialState (np.matrix): The state before the measurement.
    measuredOutSystems (List[float]): The list of systems being measured out.
    entanglementMeasure (Callable[[np.matrix], float]): The entanglement measure being used in the Localizable Multipartite Entanglement.

    Returns:
    List[float]: The list containing the Localizable Multipartite Entanglements.
    """

    nMeasurements = len(measureOutSystems)  # Number of measured out systems
    totalSystems = int(np.log2(len(initialState))) #Number of systems in the state
    binaryVectors = list(map(list, itertools.product([0, 1], repeat=nMeasurements)))
    nCombinations = len(binaryVectors)

    LMEs = []

    for paramSet in range(len(measParamsMatrix[:,0])):  # iterate through the rows of the matrix
        measurementParams = measParamsMatrix[paramSet, :]
        
        avgEntanglementMeasure = 0
   
        for binVec in range(nCombinations):
            
            projector = 1
            measureOutPos = 0
            rot = 0
            
            for system in range(1, totalSystems+1):
                if measureOutSystems.count(system) == 0:
                    projector = np.kron(projector, I)
                else:
                    if binaryVectors[binVec][measureOutPos] == 0:
                        rot = U(measurementParams[4*measureOutPos:4*(measureOutPos + 1)])  # select the parameters corresponding to the correct subsystems
                        projector = np.kron(projector, zeroState.getH() @ rot.getH())
                        measureOutPos += 1
                    else:
                        rot = U(measurementParams[4*measureOutPos:4*(measureOutPos + 1)])
                        projector = np.kron(projector, oneState.getH() @ rot.getH())
                        measureOutPos += 1

            projectedState = projector @ initialState
            prob = abs((projectedState.getH() @ projectedState).item((0,0)))

            if prob > 10**(-10):
                measOutcome = projectedState / np.sqrt(prob)
                avgEntanglementMeasure += prob * entanglementMeasure(measOutcome@measOutcome.getH())  # use the passed function here

        LMEs.append(-1 * avgEntanglementMeasure)
        
    return LMEs

########################################################################################################################################################################################

def optimizedLME(initialState: np.matrix, measureOutSystems: List[float], entanglementMeasure: Callable[[np.matrix], float], part, it) -> float:

    """
    This function optmizes the Localizable Multipartite Entanglement for single qubit measurements given the pre-measurement state, the measured out systems and the entanglement
    measure.

    Parameters:
    initialState (np.matrix): The initial (pre-measurement) state of the system.
    measuredOutSystems (List[float]): The list of systems being measured out.
    entanglementMeasure (Callable[[np.matrix], float]): The entanglement measure being used in the Localizable Multipartite Entanglement.
    part: The number of particles used in the particle swarm optimization.
    it: The number of iterations used in the particle swarm optimization.

    Returns:
    float: The optimal Localizable Multipartite Entanglement.
    """

    nMeasurements = len(measureOutSystems)
    lowerBounds = np.zeros(4*nMeasurements)
    upperBounds = 2*pi*np.ones(4*nMeasurements)
    bounds = (lowerBounds, upperBounds)

    #c1 and c2 are the social and cognitive coefficient (maybe not in that order, idk for sure) for particle swarm optimization
    options = {'c1': 0.3, 'c2': 0.3, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles = part, dimensions = 4*nMeasurements, options = options, bounds = bounds)
    entDist, optimalParams = optimizer.optimize(LME, iters = it, initialState = initialState, measureOutSystems = measureOutSystems, entanglementMeasure = entanglementMeasure,verbose = False)

    return -1*entDist

########################################################################################################################################################################################

def opt(measParamsMatrix: List[np.array], initialState: np.matrix, measuredOutSystems: List[float], entanglementMeasure: Callable[[np.matrix], float]) -> List[float]:

    M = len(measuredOutSystems)
    
    ensamble = generatePostMeasurementEnsamble(measuredOutSystems, initialState, [U(measParamsMatrix[m, :]) for m in range(M)])

    imax = 2**M

    average = []

    for i in range(imax):
        average.append(-1*ensamble[i][0]*entanglementMeasure(ensamble[i][1]@ensamble[i][1].getH()))
        
    return sum(average)

########################################################################################################################################################################################

def optLoop(initialState: np.matrix, measuredOutSystems: List[float], entanglementMeasure: Callable[[np.matrix], float]) -> float:

    """
    This function optmizes the Localizable Multipartite Entanglement for single qubit measurements given the pre-measurement state, the measured out systems and the entanglement
    measure.

    Parameters:
    initialState (np.matrix): The initial (pre-measurement) state of the system.
    measuredOutSystems (List[float]): The list of systems being measured out.
    entanglementMeasure (Callable[[np.matrix], float]): The entanglement measure being used in the Localizable Multipartite Entanglement.

    Returns:
    float: The optimal Localizable Multipartite Entanglement.
    """

    nMeasurements = len(measuredOutSystems)
    lowerBounds = np.zeros(4*nMeasurements)
    upperBounds = 2*pi*np.ones(4*nMeasurements)
    bounds = (lowerBounds, upperBounds)

    #c1 and c2 are the social and cognitive coefficient (maybe not in that order, idk for sure) for particle swarm optimization
    options = {'c1': 0.3, 'c2': 0.3, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles = 10, dimensions = 4*nMeasurements, options = options, bounds = bounds)
    entDist, optimalParams = optimizer.optimize(opt, iters = 1000, initialState = initialState, measuredOutSystems = measuredOutSystems, entanglementMeasure = entanglementMeasure,verbose = False)

    return -1*entDist

########################################################################################################################################################################################

def opSqrt(operator: np.matrix) -> np.matrix:

    """
    This function calculates the square root on an operator.

    Parameters:
    operator (np.matrix): The operator.

    Returns:
    np.matrix: The square root of the operator.
    """
    
    nEvals = (np.linalg.eig(operator)[0]).size
    
    opSqrtSum = 0 

    for i in range(nEvals):
        lambdai = np.linalg.eig(operator)[0][i]
        ei = (np.linalg.eig(operator)[1])[:,i]
        opSqrtSum += np.sqrt(lambdai)*ei@ei.getH()

    return opSqrtSum    

def fidelity(densityMatrix1: np.matrix, densityMatrix2: np.matrix) -> float:

    """
    This function calculates the fidelity between two states.

    Parameters:
    densityMatrix1 (np.matrix): The first state.
    densityMatrix2 (np.matrix): The second state.
    
    Returns:
    float: The fidelity between the states.
    """    
    
    return abs(np.trace(opSqrt(opSqrt(densityMatrix2)@densityMatrix1@opSqrt(densityMatrix2))))

def frantzeskakisLE(initialState: np.matrix, measureOutSystems: List[float], phi: float, entanglementMeasure: Callable[[np.matrix], float]) -> float:
    
    """
    This function calculates the Localizable Multipartite Entanglement of a given entanglement measure given the a list of parameters for single-qubit measurements, an initial state,
    the measured out systems and the entanglement measure.

    Parameters:
    measParamsMatrix (np.array): The matrix with the parameters of the single qubit measaurements.
    initialState (np.matrix): The state before the measurement.
    measuredOutSystems (List[float]): The list of systems being measured out.
    entanglementMeasure (Callable[[np.matrix], float]): The entanglement measure being used in the Localizable Multipartite Entanglement.

    Returns:
    List[float]: The list containing the Localizable Multipartite Entanglements.
    """

    nMeasurements = len(measureOutSystems)  # Number of measured out systems
    totalSystems = int(np.log2(len(initialState))) #Number of systems in the state
    binaryVectors = list(map(list, itertools.product([0, 1], repeat=nMeasurements)))
    nCombinations = len(binaryVectors)

    avgEntanglementMeasure=0

    for binVec in range(nCombinations):
        projector = 1
        measureOutPos = 0
        rot = 0
        for system in range(1, totalSystems+1):
            if measureOutSystems.count(system) == 0:
                projector = np.kron(projector, I)
            else:
                if binaryVectors[binVec][measureOutPos] == 0:
                    eVec = np.matrix([[1], [np.exp(1j*phi)]])*1/np.sqrt(2)
                    projector = np.kron(projector, eVec.getH())
                    measureOutPos += 1
                else:
                    eVec = np.matrix([[1], [-np.exp(1j*phi)]])*1/np.sqrt(2)
                    projector = np.kron(projector, eVec.getH())
                    measureOutPos += 1

        projectedState = projector@initialState

        prob = projectedState.getH()@projectedState
        prob = abs(prob.item((0,0)))
                                
        if prob > 10**(-15):
            measOutcome = projectedState/np.sqrt(prob)
            avgEntanglementMeasure += prob * entanglementMeasure(measOutcome@measOutcome.getH())
        
    return avgEntanglementMeasure

def optLoopParams(initialState: np.matrix, measuredOutSystems: List[float], entanglementMeasure: Callable[[np.matrix], float]) -> float:

    """
    This function optmizes the Localizable Multipartite Entanglement for single qubit measurements given the pre-measurement state, the measured out systems and the entanglement measure.

    Parameters:
    initialState (np.matrix): The initial (pre-measurement) state of the system.
    measuredOutSystems (List[float]): The list of systems being measured out.
    entanglementMeasure (Callable[[np.matrix], float]): The entanglement measure being used in the Localizable Multipartite Entanglement.

    Returns:
    float: The optimal Localizable Multipartite Entanglement.
    """

    nMeasurements = len(measuredOutSystems)
    lowerBounds = np.zeros(2*nMeasurements)
    upperBounds = 2*pi*np.ones(2*nMeasurements)
    bounds = (lowerBounds, upperBounds)

    #c1 and c2 are the social and cognitive coefficient (maybe not in that order, idk for sure) for particle swarm optimization
    options = {'c1': 0.3, 'c2': 0.3, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles = 10, dimensions = 2*nMeasurements, options = options, bounds = bounds)
    entDist, optimalParams = optimizer.optimize(optParams, iters = 1000, initialState = initialState, measuredOutSystems = measuredOutSystems, entanglementMeasure = entanglementMeasure,verbose = False)

    return -1*entDist


########################################################################################################################################################################################


def qr_haar(N):
    
    """
    Generates an NxN Haar-random matrix using the QR decomposition.
    Implementation from https://pennylane.ai/qml/demos/tutorial_haar_measure.

    Parameters:
    N (int): The size of the matrix.

    Returns:
    np.matrix: The Haar-random matrix.
    
    """
    
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    Q, R = np.linalg.qr(Z)

    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    return np.dot(Q, Lambda)




