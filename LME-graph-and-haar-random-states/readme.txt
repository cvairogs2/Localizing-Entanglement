The code here may be used to compute the LME of arbitrary states specified by their state vector.

To generate a state vector (matrix) corresponding to (weighted) graph state as in sections 7 and 8 of the main article, use the "generateGraphState" function.

To generate a Haar random state as in section 3 of the main article, use the "qr_Haar" function to generate a Haar-random unitary and multiply it by a fixed state vector. 

To directly compute the LME of a state vector, use the "optimizedLME" function and specify the entanglement measure of choice. Similar functions such as "opt," "optLoop," and "LME" are supporting functions used in the particle swarm optimization that computes the LME. 

Other relevant functions may be found with commented explanations in the python file. 



Example 1: Computes the LME as defined by the n-tangle and its upper bound from Theorem 1 for a 4-qubit Haar-random state with qubits 2 and 4 measured out. Here, 10 particles and 100 iterations are used in the particle swarm optimization. 

> state = qr_haar(2**4)@np.kron(np.kron(np.kron(zeroState, zeroState), zeroState), zeroState)
> LE = optimizedLME(state, [2,4], nTangle, 10, 100)
> psiB = partialTrace([2,4], state@state.getH())
> psiBTilde = YProd(2)@psiB.conjugate()@YProd(2)
> UB = F(psiB, psiBTilde)
> print(LE, UB)



Example 2: Computes the LME as defined by the concentratable entanglement (CE) with s being the full-qubit set of the post-measurement state. The input state is a 4-qubit ring graph and qubits 1,2 are measured out. The upper bound from Theorem 3 is also computed. Note that because the CE has an implicit dependence on s, we cannot simply input the "CE" function into "optimizedLE" to compute the LME. Thus, we specify the entanglement measure with a lambda function to simultaneously input the set s.    

> adjacency = np.array([[0, 1, 0, 1],
>                       [1, 0, 1, 0],
>                       [0, 1, 0, 1],
>                       [1, 0, 1, 0]])
> state = generateGraphState(adjacency, 0.9*pi)
> LE = optimizedLME(state, [1,2], lambda densityMatrix: CE(densityMatrix, [1,2]), 10, 100)
> UB = CE(state@state.getH(), [3,4])
> print(LE, UB)
