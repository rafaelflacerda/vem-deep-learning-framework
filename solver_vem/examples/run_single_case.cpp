#include <ios>
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void printSection(const std::string& title) {
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '-') << "\n";
}

void printMatrix(const MatrixXd& mat, const std::string& name, bool compact = false) {
    std::cout << name << " [" << mat.rows() << "x" << mat.cols() << "]\n";

    if (compact && (mat.rows() > 10 || mat.cols() > 10)) {
        std::cout << "(Matriz grande - mostrando apenas dimensões)\n";
        std::cout << "  Primeiras 3 linhas e 3 colunas:\n";
        int show_rows = std::min(3, (int)mat.rows());
        int show_cols = std::min(3, (int)mat.cols());
        for (int i = 0; i < show_rows; i++) {
            std::cout << "  ";
            for (int j = 0; j < show_cols; j++) {
                std::cout << std::setw(14) << mat(i,j) << " ";
            }
            if (show_cols < mat.cols()) std::cout << "...";
            std::cout << "\n";
        }
        if (show_rows < mat.rows()) std::cout << "  ...\n";
    } else {
        std::cout << mat << "\n";
    }
}

void printVector(const VectorXd& vec, const std::string& name, bool compact = false) {
    std::cout << name << " [" << vec.size() << "]\n";

    if (compact && vec.size() > 20) {
        std::cout << "(Vetor grande - mostrando primeiros 5 e últimos 5 elementos)\n";
        std::cout << "  ";
        for (int i = 0; i < 5; i++) {
            std::cout << vec(i);
            if (i < 4) std::cout << ", ";
        }
        std::cout << " ... ";
        for (int i = vec.size() - 5; i < vec.size(); i++) {
            std::cout << vec(i);
            if (i < vec.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
    } else {
        std::cout << "  ";
        for (int i = 0; i < vec.size(); i++) {
            std::cout << vec(i);
            if (i < vec.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

int main()
{
    std::cout << std::scientific << std::setprecision(2);

    printHeader("VEM 1D Beam Analysis - Debug Mode");

    // Configuração
    const double L = 1.0;
    const int N_ELEMENTS = 21;
    const double Q0 = -1.0;
    const int ORDER = 4;

    std::cout << "\nParâmetros da Simulação:\n";
    std::cout << "  Comprimento da viga: " << L << " m\n";
    std::cout << "  Número de elementos: " << N_ELEMENTS << "\n";
    std::cout << "  Ordem do modelo: " << ORDER << "\n";
    std::cout << "  Carga distribuída: " << Q0 << " N/m\n";

    // Discretização
    mesh::beam bar;
    bar.horizontalBarDisc(L, N_ELEMENTS);

    MatrixXd nodes = bar.nodes;
    MatrixXi elements = bar.elements;

    VectorXd q = VectorXd::Zero(2);
    q(0) = Q0;
    q(1) = Q0;

    material::mat elastic;
    elastic.setElasticModule(2.1e+11);
    double E = elastic.E;

    solver::beam1d solver(nodes, elements, ORDER);
    solver.setInertiaMoment((0.02 * std::pow(0.003, 3)) / 12.0);

    // Montagem da matriz de rigidez
    printSection("1. Montagem da Matriz de Rigidez Global");
    MatrixXd K = solver.buildGlobalK(E);
    printMatrix(K, "K (antes da condensação)", true);

    // Condensação estática - sub-blocos
    printSection("2. Particionamento para Condensação Estática");
    MatrixXd KII = solver.buildStaticCondensation(K, "KII");
    MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
    MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
    MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");

    std::cout << "\nDimensões dos sub-blocos:\n";
    std::cout << "  KII (Nodal):      " << KII.rows() << "x" << KII.cols() << "\n";
    std::cout << "  KMM (Momentos):   " << KMM.rows() << "x" << KMM.cols() << "\n";
    std::cout << "  KIM (Acoplamento): " << KIM.rows() << "x" << KIM.cols() << "\n";

    // Vetor de carga
    printSection("3. Montagem do Vetor de Carga Global");
    solver.setDistributedLoad(q, elements);
    VectorXd R = solver.buildGlobalDistributedLoad();
    printVector(R, "R (antes da condensação)", true);

    VectorXd RI = solver.buildStaticDistVector(R, "RI");
    VectorXd RM = solver.buildStaticDistVector(R, "RM");

    std::cout << "\nDimensões dos sub-vetores:\n";
    std::cout << "  RI (Nodal):   " << RI.size() << "\n";
    std::cout << "  RM (Momentos): " << RM.size() << "\n";

    // Condensação final
    printSection("4. Condensação Estática");
    MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
    VectorXd R_ = RI - KIM * KMM.inverse() * RM;

    std::cout << "Sistema condensado:\n";
    std::cout << "  K_: " << K_.rows() << "x" << K_.cols() << "\n";
    std::cout << "  R_: " << R_.size() << "\n";

    // Condições de contorno
    printSection("5. Aplicação de Condições de Contorno");
    MatrixXi supp = MatrixXi::Zero(1,4);
    supp(0,0) = 0;
    supp(0,1) = 1;
    supp(0,2) = 1;
    supp(0,3) = 0;
    solver.setSupp(supp);

    std::cout << "Restrições aplicadas no nó " << supp(0,0) << ":\n";
    std::cout << "  Deslocamento w: " << (supp(0,1) ? "restrito" : "livre") << "\n";
    std::cout << "  Rotação θ: " << (supp(0,2) ? "restrita" : "livre") << "\n";

    K_ = solver.applyDBCMatrix(K_);
    R_ = solver.applyDBCVec(R_);

    // Solução
    printSection("6. Solução do Sistema Linear");
    VectorXd uh = K_.ldlt().solve(R_);

    std::cout << "\nResultados:\n";
    std::cout << "  DOFs totais: " << uh.size() << "\n";
    std::cout << "  Deslocamento máximo: " << uh.maxCoeff() << " m\n";
    std::cout << "  Deslocamento mínimo: " << uh.minCoeff() << " m\n";

    printSection("7. Vetor Solução Completo");
    printVector(uh, "u_h", true);

    printHeader("Simulação Concluída");

    return 0;
}
