#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>
#include <map>

using namespace std;
using namespace Eigen;

// Define the Pauli matrices
Matrix2cd getPauliMatrix(int index) {
    Matrix2cd X, Y, Z, I;
    X << 0, 1,
         1, 0;
    Y << 0, complex<double>(0, -1),
         complex<double>(0, 1), 0;
    Z << 1, 0,
         0, -1;
    I << 1, 0,
         0, 1;

    switch (index) {
        case 0: return X;
        case 1: return Y;
        case 2: return Z;
        case 3: return I;
        default: return I;
    }
}

double twoPointQ(int N, const VectorXcd& state, const vector<int>& indexes) {
    Matrix3cd Q = Matrix3cd::Zero();
    int dim = 1 << N;  // 2^N, the dimension of the Hilbert space

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            MatrixXcd op1 = MatrixXcd::Identity(1, 1);
            MatrixXcd op2 = MatrixXcd::Identity(1, 1);
            MatrixXcd op3 = MatrixXcd::Identity(1, 1);

            int count = 0;

            for (int k = 0; k < N; ++k) {
                Matrix2cd pauliMatrix = getPauliMatrix(3);  // Start with Identity
                if (find(indexes.begin(), indexes.end(), k + 1) != indexes.end()) {
                    if (count == 0) {
                        op1 = kroneckerProduct(op1, getPauliMatrix(i)).eval();
                        op2 = kroneckerProduct(op2, getPauliMatrix(i)).eval();
                        op3 = kroneckerProduct(op3, getPauliMatrix(3)).eval();
                    } else {
                        op1 = kroneckerProduct(op1, getPauliMatrix(j)).eval();
                        op2 = kroneckerProduct(op2, getPauliMatrix(3)).eval();
                        op3 = kroneckerProduct(op3, getPauliMatrix(j)).eval();
                    }
                    count++;
                }
                else{
                    op1 = kroneckerProduct(op1, getPauliMatrix(3)).eval();
                    op2 = kroneckerProduct(op2, getPauliMatrix(3)).eval();
                    op3 = kroneckerProduct(op3, getPauliMatrix(3)).eval();
                }

            }

            // Ensure that operators are now dim x dim matrices
            assert(op1.rows() == dim && op1.cols() == dim);
            assert(op2.rows() == dim && op2.cols() == dim);
            assert(op3.rows() == dim && op3.cols() == dim);

            complex<double> term1 = (state.adjoint() * op1 * state)(0, 0);
            complex<double> term2 = (state.adjoint() * op2 * state)(0, 0);
            complex<double> term3 = (state.adjoint() * op3 * state)(0, 0);

            Q(i, j) = term1 - term2 * term3;
        }
    }

    // Compute the SVD of Q without using ComputeThinU | ComputeThinV
    JacobiSVD<Matrix3cd> svd(Q);
    Vector3d singularValues = svd.singularValues();

    return singularValues(0);  // The maximum singular value
}

int main() {
    // Example usage of the twoPointQ function

    int N = 3; // Number of qubits
    VectorXcd state(8);  // Quantum state vector (dimension 2^N = 4 for N=2)
    state << 707, 0, 0, 0, 0, 0, 707, 0;  // Example state (|00>)

    vector<vector<int>> indexes = {{1, 2}, {1, 3}, {2, 3}};  // Indexes of qubits

    vector<double> maxSingularValues;

    for (const auto& loop : indexes){
        maxSingularValues.push_back(twoPointQ(N, state, loop));
    }

    cout << "Maximum singular value: " << *max_element(maxSingularValues.begin(), maxSingularValues.end()) << endl;

    for (const auto& member: maxSingularValues){
        cout << member << endl;
    }
    
    return 0;
}