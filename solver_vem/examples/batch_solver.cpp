#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Uso: batch_solver <case_id> <L> <q> <E> <I> <n_elements>" << std::endl;
        return 1;
    }

    std::string case_id = argv[1];
    double L = std::stod(argv[2]);
    double q_val = std::stod(argv[3]);
    double E = std::stod(argv[4]);
    double I = std::stod(argv[5]);
    int n_elements = std::stoi(argv[6]);

    mesh::beam bar;
    bar.horizontalBarDisc(L, n_elements);

    Eigen::MatrixXd nodes = bar.nodes;
    Eigen::MatrixXi elements = bar.elements;

    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    q(0) = -q_val;
    q(1) = -q_val;

    material::mat elastic;
    elastic.setElasticModule(E);

    int order = 3;
    solver::beam1d solver(nodes, elements, order);
    solver.setInertiaMoment(I);

    Eigen::MatrixXd K = solver.buildGlobalK(E);

    Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
    Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
    Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
    Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");

    solver.setDistributedLoad(q, elements);
    Eigen::VectorXd R = solver.buildGlobalDistributedLoad();
    Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
    Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");

    Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1, 4);
    supp(0, 0) = 0;
    supp(0, 1) = 1;
    supp(0, 2) = 1;
    supp(0, 3) = 0;
    solver.setSupp(supp);

    Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
    Eigen::VectorXd R_ = RI - KIM * KMM.inverse() * RM;

    K_ = solver.applyDBCMatrix(K_);
    R_ = solver.applyDBCVec(R_);

    Eigen::VectorXd uh = K_.ldlt().solve(R_);

    int n_nodes = uh.size() / 2;

    std::cout << std::setprecision(17);
    std::cout << case_id << "|" << n_elements << "|";

    for (int i = 0; i < n_nodes; ++i) {
        if (i > 0) std::cout << ";";
        std::cout << uh(2 * i);
    }

    std::cout << "|";

    for (int i = 0; i < n_nodes; ++i) {
        if (i > 0) std::cout << ";";
        std::cout << uh(2 * i + 1);
    }

    std::cout << std::endl;

    return 0;
}
