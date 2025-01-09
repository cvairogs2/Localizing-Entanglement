#include "QuantumObjects.h"
#include <iostream>
#include <pagmo/pagmo.hpp>
#include <vector>
#include <tuple>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace pagmo;

QuantumObjects qo;

// Define the objective function for n dimensions
struct optimizer {
    // The dimension of the problem, static to set before instantiation
    static double angle;
    static Matrix<int, Dynamic, Dynamic> adjacencyMatrix;
    static vector<int> measuredOutSystems;

    int nMeasurements = measuredOutSystems.size();
    int dimensions = 4*nMeasurements;

    // Function to calculate the objective
    vector<double> fitness(const vector<double> &x) const {

        vector<MatrixXcd> measurements;

        for (int i = 0; i < nMeasurements; i++){
            measurements.push_back(qo.U(x[i*nMeasurements], x[i*nMeasurements+1], x[i*nMeasurements+2], x[i*nMeasurements+3]));
        };

        vector<tuple<double, MatrixXcd>> ensemble = qo.postMeasEns(measuredOutSystems, measurements, qo.generateGraphState(adjacencyMatrix, angle));

        double obj = 1.0; // Start with 1

        for (const auto& member : ensemble) {
            obj -= get<0>(member)*qo.concentratableEntanglement(get<1>(member)*get<1>(member).adjoint(), {2, 3});
        }

        return {obj}; // Return the objective in a vector
    }

    // Provide the number of dimensions in the problem
    pair<vector<double>, vector<double>> get_bounds() const {
        vector<double> lb(dimensions, 0); // Lower bounds of 0 for each dimension
        vector<double> ub(dimensions, 2*pi);  // Upper bounds of 2*pi for each dimension
        return {lb, ub};
    }
};

int nQubits = 4; // Determine the size of the system for the adjacency matrix

double optimizer::angle;
Matrix<int, Dynamic, Dynamic> optimizer::adjacencyMatrix;
vector<int> optimizer::measuredOutSystems; // Define the static member outside the struct

int main(){

    int dim = 1 << nQubits; // Determine the dimension of the Hilbert Space
    // Define an adjacency matrix for a simple graph
    MatrixXi adjM = MatrixXi::Zero(nQubits, nQubits);

    for (int i=0; i < nQubits; i++) {
        for (int j=0; j < nQubits; j++){
            if (i-j==1 || j-i==1){
                adjM(i, j) = 1;
            }
        }
    };

    vector<int> measOutSystems = {2, 3};

    optimizer::adjacencyMatrix = adjM;
    optimizer::measuredOutSystems = measOutSystems;
    int count = 1;

    // Looping for angles

    int mMax = 29;

    for (int m=0; m <= mMax; m++) {

        optimizer::angle = (m*pi)/mMax;

        problem prob{optimizer()}; // Initialize the problem
        algorithm algo{pso(10)}; // Use PSO with 100 generations
        population pop{prob,10}; // Create a population of 100 particles
        // Evolve the population to optimize the function
        pop = algo.evolve(pop);
        // Extract and display the best result
        auto best = pop.champion_x();

        cout << count << " " << (m*pi)/mMax << " " << 1.0-pop.champion_f()[0] << endl;

        count++;

    };

	return 0;
}