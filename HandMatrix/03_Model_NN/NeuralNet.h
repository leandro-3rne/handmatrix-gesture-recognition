#ifndef NEURALNET_H
#define NEURALNET_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>

using namespace Eigen;

class NeuralNet {
private:
    MatrixXd W1, W2, W3;
    VectorXd b1, b2, b3;
    // Sigmoid braucht mehr "Schwung" als ReLU, daher höhere Lernrate!
    double learningRate = 0.2;

public:
    // Konstruktor
    NeuralNet(int inputSize, int hidden1Size, int hidden2Size, int outputSize) {
        // Klassische Initialisierung für Sigmoid (Werte zwischen -1 und 1)
        W1 = MatrixXd::Random(hidden1Size, inputSize);
        b1 = VectorXd::Random(hidden1Size);

        W2 = MatrixXd::Random(hidden2Size, hidden1Size);
        b2 = VectorXd::Random(hidden2Size);

        W3 = MatrixXd::Random(outputSize, hidden2Size);
        b3 = VectorXd::Random(outputSize);

        std::cout << "Neural Net (Sigmoid Edition) initialisiert!" << std::endl;
    }

    // --- DIE SANFTE KURVE ---
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Ableitung: x * (1 - x)
    // WICHTIG: Wir nehmen an, dass 'x' hier schon das Ergebnis von Sigmoid ist!
    double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    // --- FORWARD PASS (Denken) ---
    VectorXd forward(VectorXd input) {
        // Alles läuft durch Sigmoid -> Zahlen bleiben immer brav zwischen 0 und 1
        VectorXd h1 = (W1 * input + b1).unaryExpr([&](double x){ return sigmoid(x); });
        VectorXd h2 = (W2 * h1 + b2).unaryExpr([&](double x){ return sigmoid(x); });
        VectorXd out = (W3 * h2 + b3).unaryExpr([&](double x){ return sigmoid(x); });
        return out;
    }

    // --- TRAINING (Lernen) ---
    void train(VectorXd input, VectorXd target) {

        // 1. Forward Pass (Erinnerung auffrischen)
        VectorXd h1 = (W1 * input + b1).unaryExpr([&](double x){ return sigmoid(x); });
        VectorXd h2 = (W2 * h1 + b2).unaryExpr([&](double x){ return sigmoid(x); });
        VectorXd out = (W3 * h2 + b3).unaryExpr([&](double x){ return sigmoid(x); });

        // 2. Fehler am Ausgang
        VectorXd outputError = target - out;
        VectorXd outputGradient = outputError.array() * out.unaryExpr([&](double x){ return sigmoidDerivative(x); }).array();

        // 3. Fehler zurück zu Hidden 2
        VectorXd h2Error = W3.transpose() * outputGradient;
        VectorXd h2Gradient = h2Error.array() * h2.unaryExpr([&](double x){ return sigmoidDerivative(x); }).array();

        // 4. Fehler zurück zu Hidden 1
        VectorXd h1Error = W2.transpose() * h2Gradient;
        VectorXd h1Gradient = h1Error.array() * h1.unaryExpr([&](double x){ return sigmoidDerivative(x); }).array();

        // 5. Gewichte anpassen
        W3 += learningRate * outputGradient * h2.transpose();
        b3 += learningRate * outputGradient;

        W2 += learningRate * h2Gradient * h1.transpose();
        b2 += learningRate * h2Gradient;

        W1 += learningRate * h1Gradient * input.transpose();
        b1 += learningRate * h1Gradient;
    }

    // Optional: Lernrate von außen ändern
    void setLearningRate(double lr) {
        learningRate = lr;
    }
};

#endif //NEURALNET_H