#ifndef QUANTUMOBJECTS_H
#define QUANTUMOBJECTS_H

#include <iostream>
#include <complex>
#include <functional>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <vector>
#include <bitset>
#include <algorithm>
#include <cmath>
#include <tuple>

using namespace Eigen;
using namespace std;

// Define constants
const double pi = 3.14159265358979323846;
const double e = 2.71828182845904523536;
const int nqubits = 5;
const int dimensions = static_cast<int>(pow(2, nqubits));


struct QuantumObjects {
    Matrix<complex<double>, 2, 2> X, Y, Z, I, H, dm0, dm1, dmp, dmm;
    Matrix<complex<double>, 2, 1> st0, st1, stp, stm;
    function<Matrix<complex<double>, 2, 1>(const int x)> stx;
    function<Matrix<complex<double>, Dynamic, Dynamic>(int, int, int, double)> CP;
    function<Matrix<complex<double>, 2, 2>(double)> S;
    function<Matrix<complex<double>, 2, 2>(double, double, double, double)> U;
    function<Matrix<complex<double>, 2, 2>(double, double)> M;
    function<Matrix<complex<double>, Dynamic, Dynamic>(int, double)> SGate;
    function<Matrix<complex<double>, Dynamic, Dynamic>(int, double, double, double, double)> UGate;
    function<Matrix<complex<double>, Dynamic, 1>(const string&)> generateState;
    function<Matrix<complex<double>, Dynamic, 1>(const int&)> plusSuper, ghzState;
    function<Matrix<complex<double>, Dynamic, Dynamic>(const MatrixXi&, double)> generateGraphState;
    function<Matrix<complex<double>, Dynamic, Dynamic>(const vector<int>&, const MatrixXcd&)> partialTrace;
    function<vector<tuple<double, MatrixXcd>>(const vector<int>&, const vector<MatrixXcd>&, const MatrixXcd&)> postMeasEns;
    function<double(const MatrixXcd&)> nTangle, gmeConcurrence;
    function<double(const MatrixXcd&, const vector<int>&)> concentratableEntanglement;

    QuantumObjects() {
        // Define the complex unit i
        complex<double> i(0, 1);

        // Initialize Pauli matrices
        X << 0, 1, 1, 0;
        Y << 0, -i, i, 0;
        Z << 1, 0, 0, -1;
        I << 1, 0, 0, 1;

        // Initialize the Hadamard gate
        double sqrt2_inv = 1.0 / sqrt(2.0);
        H << sqrt2_inv, sqrt2_inv, sqrt2_inv, -sqrt2_inv;

        // Initialize Density Matices
        dm0 << 1, 0, 0, 0;
        dm1 << 0, 0, 0, 1;
        dmp << 0.5, 0.5, 0.5, 0.5;
        dmm << 0.5, -0.5, -0.5, 0.5;

        // Initialize states |0>, |1>, |+>, and |->
        st0 << 1, 0;
        st1 << 0, 1;
        stp << sqrt2_inv, sqrt2_inv;
        stm << sqrt2_inv, -sqrt2_inv;

        stx = [](int x) -> Matrix<complex<double>, 2, 1>{
            Matrix<complex<double>, 2, 1> state;
            state << 1-x, x;
            return state;
        };

        // Lambda for the S(phase) operator
        S = [i](double phi) -> Matrix<complex<double>, 2, 2> {
            Matrix<complex<double>, 2, 2> result;
            result << 1, 0, 0, exp(i * phi);
            return result;
        };

        // Initialize the lambda function for U
        U = [i](double alpha, double beta, double gamma, double delta) -> Matrix<complex<double>, 2, 2> {
            Matrix<complex<double>, 2, 2> result;

            complex<double> exp_alpha_minus_beta_delta_2 = exp(i*(alpha - beta / 2 - delta / 2));
            complex<double> exp_alpha_minus_beta_plus_delta_2 = exp(i*(alpha - beta / 2 + delta / 2));
            complex<double> exp_alpha_plus_beta_minus_delta_2 = exp(i*(alpha + beta / 2 - delta / 2));
            complex<double> exp_alpha_plus_beta_plus_delta_2 = exp(i*(alpha + beta / 2 + delta / 2));

            result(0, 0) = exp_alpha_minus_beta_delta_2 * cos(gamma / 2);
            result(0, 1) = -exp_alpha_minus_beta_plus_delta_2 * sin(gamma / 2);
            result(1, 0) = exp_alpha_plus_beta_minus_delta_2 * sin(gamma / 2);
            result(1, 1) = exp_alpha_plus_beta_plus_delta_2 * cos(gamma / 2);

            return result;
        };

        // Initialize the lambda function for V
        M = [i](double theta, double phi) -> Matrix<complex<double>, 2, 2> {
            Matrix<complex<double>, 2, 2> result;

            complex<double> exphi = exp(i*phi);
            complex<double> exmphi = exp(-i*phi);

            result(0, 0) = cos(theta / 2);
            result(0, 1) = exmphi * sin(theta / 2);
            result(1, 0) = -exphi * sin(theta / 2);
            result(1, 1) = cos(theta / 2);

            return result;
        };

        // Lambda for the S(phase) gate
        SGate = [this](int n, double phi) -> Matrix<complex<double>, Dynamic, Dynamic> {
            Matrix<complex<double>, Dynamic, Dynamic> product = Matrix<complex<double>, 1, 1>::Ones();
            for (int m=1; m <= nqubits; m++){
                if(m==n){
                    product = kroneckerProduct(product, S(phi)).eval();                    
                }
                else{
                    product = kroneckerProduct(product, I).eval();
                }
            }
            return product;
        };

        // Initialize the lambda function for U
        UGate = [this](int n, double alpha, double beta, double gamma, double delta) -> Matrix<complex<double>, Dynamic, Dynamic> {
            Matrix<complex<double>, Dynamic, Dynamic> product = Matrix<complex<double>, 1, 1>::Ones();

            for (int m=1; m <= nqubits; m++){
                if(m==n){
                    product = kroneckerProduct(product, U(alpha, beta, gamma, delta)).eval();                    
                }
                else{
                    product = kroneckerProduct(product, I).eval();
                }
            }

            return product;
        };

        // Lambda for CP(phase) operator
        CP = [this](int n, int c, int t, double phi) -> Matrix<complex<double>, Dynamic, Dynamic> {
            Matrix<complex<double>, Dynamic, Dynamic> firstTerm = Matrix<complex<double>, 1, 1>::Ones();
            Matrix<complex<double>, Dynamic, Dynamic> secondTerm = Matrix<complex<double>, 1, 1>::Ones();

            for (int m = 1; m <= n; ++m) { // Note: Loop should start from 1 and go up to n inclusive
                if (m == c) {
                    firstTerm = kroneckerProduct(firstTerm, dm0).eval();
                    secondTerm = kroneckerProduct(secondTerm, dm1).eval();
                } else if (m == t) {
                    firstTerm = kroneckerProduct(firstTerm, I).eval();
                    secondTerm = kroneckerProduct(secondTerm, S(phi)).eval(); // Correct usage of S(phi)
                } else {
                    firstTerm = kroneckerProduct(firstTerm, I).eval();
                    secondTerm = kroneckerProduct(secondTerm, I).eval();
                }
            }

            auto CP = firstTerm + secondTerm;
            return CP;
        };

        // Lambda for state generation
        generateState = [this](const string& binaryString) -> Matrix<complex<double>, Dynamic, 1> {
            Matrix<complex<double>, Dynamic, 1> stateVector = (binaryString[0] == '0') ? st0 : st1;
            for (size_t i = 1; i < binaryString.size(); i++) {
                Matrix<complex<double>, Dynamic, 1> nextQubit = (binaryString[i] == '0') ? st0 : st1;
                stateVector = kroneckerProduct(stateVector, nextQubit).eval();
            }
            return stateVector;
        };

        // Lambda for generating |+>^n superposition
        plusSuper = [this, sqrt2_inv](const int n) -> Matrix<complex<double>, Dynamic, 1> {
            Matrix<complex<double>, Dynamic, 1> stateVector = stp;
            for (int m = 1; m < n; m++) {
                stateVector = kroneckerProduct(stateVector, stp).eval();
            }
            return stateVector;
        };

        // Lambda for generating |+>^n superposition
        ghzState = [this, sqrt2_inv](const int n) -> Matrix<complex<double>, Dynamic, 1> {
            int dim = 1 << n;
            Matrix<complex<double>, Dynamic, 1> ghzState = MatrixXcd::Zero(dim, 1);  // Initialize a column vector with zeros
            // Assign values to the first and last elements
            ghzState(0, 0) = sqrt2_inv;
            ghzState(dim-1, 0) = sqrt2_inv;
            return ghzState;
        };

        // Lambda for generating a graph state from an adjacency matrix
        generateGraphState = [this](const MatrixXi& adjMatrix, double phi) -> Matrix<complex<double>, Dynamic, Dynamic> {
            int n = adjMatrix.rows(); // Assuming adjMatrix is square
            auto initialState = plusSuper(n); // Generate the initial superposition state |+>^n
            
            // Placeholder for resulting graph state
            Matrix<complex<double>, Dynamic, 1> graphState = initialState;

            // Apply CP gates according to the adjacency matrix
            for (int i = 0; i < n; ++i) {
                for (int j = i+1; j < n; ++j) {
                    if (adjMatrix(i, j) == 1) {
                        // Apply a CP gate between qubits i and j
                        // cout << "Applying CP gate between qubits " << i+1 << " and " << j+1 << " with phi=" << phi << endl;
                        // Note: Assuming CP returns a matrix representing the operation
                        // and graphState needs to be updated accordingly.
                        // This line conceptualizes applying the CP gate to the current graph state.
                        // Actual implementation may vary based on how CP is defined and used.
                        graphState = CP(n, i+1, j+1, phi) * graphState;
                    }
                }
            }

            return graphState;
        };
        
        // Lambda for calculating the partial trace
        partialTrace = [this](const vector<int>& tracedOutSystems, const MatrixXcd& densityMatrix) -> Matrix<complex<double>, Dynamic, Dynamic> {
            int totalSystems = static_cast<int>(log2(densityMatrix.rows()));
            int n = tracedOutSystems.size();  // Use size() instead of rows() because it's a vector now
            int totalCombinations = pow(2, n);
            int m = pow(2, totalSystems - n);
            int dim = pow(2, totalSystems);
            vector<vector<int>> combinations;

            Matrix<complex<double>, Dynamic, Dynamic> marginal = Matrix<complex<double>, Dynamic, Dynamic>::Zero(m, m);

            for (int i = 0; i < totalCombinations; i++) {
                vector<int> combination;
                for (int j = n - 1; j >= 0; j--) {
                    // Shift i by j bits to the right and check the least significant bit
                    int bit = (i >> j) & 1;
                    combination.push_back(bit);
                }
                combinations.push_back(combination);
            }

            for (int i = 0; i < totalCombinations; i++) {
                Matrix<complex<double>, Dynamic, Dynamic> prodIter = Matrix<complex<double>, 1, 1>::Ones();
                for (int j = 1; j <= totalSystems; j++) {
                    int check = 0;
                    int cond = 0;
                    while (check < n) { // this is a good check function
                        if (j == tracedOutSystems[check]) {
                            cond = 1;
                            break;
                        } else {
                            check++;
                        }
                    }

                    if (cond == 1) {
                        if (combinations[i][check] == 0) {
                            prodIter = kroneckerProduct(prodIter, st0).eval();
                        } else {
                            prodIter = kroneckerProduct(prodIter, st1).eval();
                        }
                    } else {
                        prodIter = kroneckerProduct(prodIter, I).eval();
                    }
                }
                marginal += prodIter.adjoint() * densityMatrix * prodIter;
            }

            return marginal;
        };

        // Lambda for generating the post-measurement ensamble
        postMeasEns = [this](const vector<int>& measOutSystems, const vector<MatrixXcd>& Measurements, const MatrixXcd& initialState) -> vector<tuple<double, MatrixXcd>>{
            int n = measOutSystems.size();
            int totalCombinations = pow(2, n);
            int totalSystems = static_cast<int>(log2(initialState.rows()));
            vector<vector<int>> combinations;
            vector<tuple<double, MatrixXcd>> Ens;

            for (int i = 0; i < totalCombinations; i++) {
                vector<int> combination;
                for (int j = n - 1; j >= 0; j--) {
                    // Shift i by j bits to the right and check the least significant bit
                    int bit = (i >> j) & 1;
                    combination.push_back(bit);
                }
                combinations.push_back(combination);
            }

            for (int i = 0; i < totalCombinations; i++) {
                Matrix<complex<double>, Dynamic, Dynamic> projector = Matrix<complex<double>, 1, 1>::Ones();
                Matrix<complex<double>, Dynamic, 1> projectedVec;
                Matrix<complex<double>, Dynamic, 1> measOutState;
                double prob = 0;
                for (int j = 1; j <= totalSystems; j++) {
                    bool found = false;
                    for (int check = 0; check < n; ++check) {
                        if (j == measOutSystems[check]) {
                            found = true;
                            if (combinations[i][check] == 0) {
                                projector = kroneckerProduct(projector, (Measurements[check] * st0).eval()).eval();
                            } else {
                                projector = kroneckerProduct(projector, (Measurements[check] * st1).eval()).eval();
                            }
                            break;
                        }
                    }

                    if (!found) {
                        projector = kroneckerProduct(projector, I).eval();
                    }
                }

                projectedVec = projector.adjoint() * initialState;

                prob = (projectedVec.adjoint() * projectedVec).norm();

                measOutState = projectedVec / sqrt(prob);

                Ens.push_back({prob, measOutState});
            }

            return Ens;
        };

        // Lambda for the nTangle of a quantum state
        nTangle = [this](const MatrixXcd& densityMatrix) -> double {
            int totalSystems = static_cast<int>(log2(densityMatrix.rows()));
            Matrix<complex<double>, Dynamic, Dynamic> yProd = Matrix<complex<double>, 1, 1>::Ones();

            for (int i=0; i < totalSystems; i++){
                yProd = kroneckerProduct(yProd, Y).eval();
            }

            return sqrt(abs((densityMatrix*yProd*densityMatrix.conjugate()*yProd).trace()));

        };
        
        // Lambda for the gmeConcurrence of a density matrix
        gmeConcurrence = [this](const MatrixXcd& densityMatrix) -> double {
            int dim = densityMatrix.rows();
            int totalSystems = static_cast<int>(log2(dim));
            double minConcur = 1.0;

            vector<vector<int>> allCombinations;

            // Loop through all possible binary combinations from 1 to 2^n - 2
            for (int i = 1; i < (1 << totalSystems) - 1; ++i) { // Loop from 1 to 2^n - 2 (excluding 0 and 2^n - 1)
                bitset<32> b(i); // Use 32 bits to accommodate larger n values

                // Determine the values based on the binary representation
                vector<int> values;
                for (int j = 0; j < totalSystems; ++j) {
                    if (b[j] == 1) {
                        values.push_back(j + 1); // Indexing starting from 1
                    }
                }

                allCombinations.push_back(values); // Store the current combination

            }

            for (const auto& member : allCombinations) {
                Matrix<complex<double>, Dynamic, Dynamic> marginal = partialTrace(member, densityMatrix);
                double concurrence = sqrt(2 * (1 - (marginal * marginal).trace().real()));
                if (concurrence <= minConcur) {
                    minConcur = concurrence;
                }
            }

            return minConcur;
        };

        // Lambda for the Concentratable Entanglement of a density matrix
        concentratableEntanglement = [this](const MatrixXcd& densityMatrix, const vector<int>& s) -> double {

            int subSetCardinality = s.size();
            int dim = densityMatrix.rows();
            int totalSystems = static_cast<int>(log2(dim));

            vector<vector<int>> power_set;

            // Generate the power set of the chosen subset
            for (int i = 1; i < (1 << subSetCardinality); ++i) {  // Start from 1 to exclude the empty set
                vector<int> subset;
                for (int j = 0; j < subSetCardinality; ++j) {
                    if ((i >> j) & 1) {
                        subset.push_back(s[j]);
                    }
                }
                power_set.push_back(subset);
            }

            double ce = 1.0 - pow(2.0, -subSetCardinality);
            int powerSetCardinality = power_set.size();

            for (int i = 0; i < powerSetCardinality; ++i) {
                MatrixXcd d = partialTrace(power_set[i], densityMatrix);  // Assuming partialTrace is implemented elsewhere
                ce -= pow(2.0, -subSetCardinality) * (d*d).eval().trace().real();  // Use .real() to get the real part of the trace
            }

            return ce;
        };
    }
};

#endif // QUANTUMOBJECTS_H