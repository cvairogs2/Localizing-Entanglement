#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <complex>
#include <chrono>
#include <fstream>
#include <iomanip>  // For setting file names with precision

using namespace Eigen;
using namespace std;
using namespace chrono;

// Define Pauli matrices with complex entries
Matrix2cd sigmaX(){
    Matrix2cd mat;
    mat << 0, 1,
           1, 0;
    return mat;
}

Matrix2cd sigmaY() {
    Matrix2cd mat;
    mat << 0, complex<double>(0, -1),
           complex<double>(0, 1), 0;
    return mat;
}

Matrix2cd hadamard(){
    Matrix2cd mat;
    mat << complex<double>(1/sqrt(2), 0), complex<double>(1/sqrt(2), 0), complex<double>(1/sqrt(2), 0), complex<double>(-1/sqrt(2), 0);
    return mat;
}

Matrix2cd sigmaZ() {
    Matrix2cd mat;
    mat << 1, 0,
           0, -1;
    return mat;
}

// Identity matrix
Matrix2cd identity() {
    return Matrix2cd::Identity();
}

// Function to create the transverse field Ising model Hamiltonian using an adjacency matrix
MatrixXcd transverseFieldIsingModel(const MatrixXi& adj, const long double J, const long double h, const long double hz) {
    int N = adj.rows();  // Number of spins

    // Pauli matrices
    Matrix2cd sx = sigmaX();
    Matrix2cd sz = sigmaZ();
    Matrix2cd id = identity();
    
    // Initialize Hamiltonian
    MatrixXcd H = MatrixXcd::Zero(pow(2, N), pow(2, N));

    // Interaction term: -J * sum(sigmaZ_i * sigmaZ_j)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (adj(i, j) != 0) {
                MatrixXcd term = MatrixXcd::Identity(1, 1);

                for (int k = 0; k < N; ++k) {
                    if (k == i || k == j) {
                        term = kroneckerProduct(term, sx).eval();
                    } else {
                        term = kroneckerProduct(term, id).eval();
                    }
                }

                H -= J * term;
            }
        }
    }

    // Transverse field term: -h * sum(sigmaX_i)
    for (int i = 0; i < N; ++i) {
        MatrixXcd term = MatrixXcd::Identity(1, 1);
        MatrixXcd termz = MatrixXcd::Identity(1, 1);

        for (int j = 0; j < N; ++j) {
            if (j == i) {
                term = kroneckerProduct(term, sz).eval();
                termz = kroneckerProduct(termz, sx).eval();
            } else {
                term = kroneckerProduct(term, id).eval();
                termz = kroneckerProduct(termz, id).eval();
            }
        }

        H -= (h * term + hz * termz);
    }

    return H;
}

int main(){
    // Determine the size of the system for the adjacency matrix
    // Determine the size of the system for the adjacency matrix
    int size;
    cout << "Enter the number of spins (N): ";
    cin >> size;
    // Define an adjacency matrix for a simple graph
    MatrixXi adjM = MatrixXi::Zero(size, size);

    for (int i=0; i < size; i++) {
        // Connect each node to its next neighbor
        if (i + 1 < size) {
            adjM(i, i + 1) = 1;
            adjM(i + 1, i) = 1;
            }

        // To form the ring, connect the last node to the first node, and to make a linear graph we comment from here
        if (i == size - 1) {
            adjM(i, 0) = 1;
            adjM(0, i) = 1;
            }
        // to here

    }

    // Prompt the user for coupling constants
    // long double J = 10.0;
    // long double h 100.0;
    // long double hz;

    // cout << "Enter the coupling constant (J): ";
    // cin >> J;
    // cout << "Enter the coupling constant (h): ";
    // cin >> h;
    // cout << "Enter the coupling constant (hz): ";
    // cin >> hz;

    // Measure time for solving the eigenvalue problem
    auto start = high_resolution_clock::now();


    // Sweep over lambda values from 0.1 to 2.5 in steps of 0.1
    for (long double J = 10.0; J <= 250.0; J += 10.0) {

        long double h = 100.0;
        long double hz = 1.0;

        MatrixXcd H = transverseFieldIsingModel(adjM, J, h, hz);

        // Compute and display the eigenvalues and eigenvectors
        SelfAdjointEigenSolver<MatrixXcd> solver(H);

        VectorXcd eigenvalues = solver.eigenvalues();
        MatrixXcd eigenvectors = solver.eigenvectors();

        // The lowest eigenvalue is the first one (sorted in ascending order)
        complex<double> lowestEigenvalue = eigenvalues(0);
        VectorXcd lowestEigenvector = eigenvectors.col(0);

        // Save the lowest eigenvector to a .txt file
        stringstream filename;
        long double lambda = J/h;
        filename << "tfimgs/ground_state_vector_lambda_" << fixed << setprecision(1) << lambda << ".txt";

        ofstream outfile(filename.str());
        if (outfile.is_open()) {
            for (size_t i = 0; i < lowestEigenvector.size(); ++i) {
                outfile << lowestEigenvector(i) << endl;
            }
            outfile.close();
            cout << "Ground state vector saved to " << filename.str() << endl;
        } else {
            cerr << "Error opening file for writing." << endl;
        }

        // cout << static_cast<int>(log2(H.rows())) << endl;
    }

    // cout << "Lowest eigenvalue:\n" << lowestEigenvalue << endl;
    // // cout << "Corresponding eigenvector:\n" << lowestEigenvector << endl;
    // cout << HN*lowestEigenvector << endl;

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    cout << "Took " << elapsed.count() << " seconds for " << size << " spins." << endl;    

    return 0;
}