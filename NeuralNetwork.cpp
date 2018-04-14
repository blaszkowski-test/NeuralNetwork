/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <algorithm>
#include <cmath>

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
}

void NeuralNetwork::firstTest()
{
    scalar = 3.8;

    expectation.matrix.assign({
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 1,
        0, 0, 1, 0,
        0, 1, 1, 0,
        0, 1, 1, 1,
        0, 1, 0, 1,
        0, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 1
    });

    Layer inLayer(LayerType::InputLayer);
    inLayer.inputOrActivation = new LayerMatrix(10, 10,{
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1

    });
    inLayer.weight = new LayerMatrix(10, 10);
    inLayer.weightChange = new LayerMatrix(10, 10, 0);
    inLayer.delta = new LayerMatrix(10, 10, 0);
    layersVector.push_back(std::move(inLayer));

    Layer firstHidden(LayerType::HiddenLayer);
    firstHidden.weight = new LayerMatrix(10, 4);
    firstHidden.product = new LayerMatrix(10, 10, 0);
    firstHidden.inputOrActivation = new LayerMatrix(10, 10, 0);
    firstHidden.weightChange = new LayerMatrix(10, 4, 0);
    firstHidden.delta = new LayerMatrix(10, 10, 0);
    layersVector.push_back(std::move(firstHidden));

    Layer outLayer(LayerType::OutputLayer);
    outLayer.product = new LayerMatrix(10, 4, 0);
    outLayer.inputOrActivation = new LayerMatrix(10, 4, 0);
    outLayer.delta = new LayerMatrix(10, 4, 0);

    layersVector.push_back(std::move(outLayer));
}

void NeuralNetwork::secondTest()
{
    scalar = 3.2;

    expectation.matrix.assign({
        0,
        1,
        1,
        1
    });

    Layer inLayer(LayerType::InputLayer);
    inLayer.inputOrActivation = new LayerMatrix(4, 2,{
        0, 0,
        0, 1,
        1, 0,
        1, 1
    });
    inLayer.weight = new LayerMatrix(2, 4);
    inLayer.weightChange = new LayerMatrix(2, 4, 0);
    inLayer.delta = new LayerMatrix(2, 4, 0);
    layersVector.push_back(std::move(inLayer));

    Layer firstHidden(LayerType::HiddenLayer);
    firstHidden.weight = new LayerMatrix(4, 4);
    firstHidden.product = new LayerMatrix(4, 4, 0);
    firstHidden.inputOrActivation = new LayerMatrix(4, 4, 0);
    firstHidden.weightChange = new LayerMatrix(4, 4, 0);
    firstHidden.delta = new LayerMatrix(4, 4, 0);
    layersVector.push_back(std::move(firstHidden));

    Layer secondHidden(LayerType::HiddenLayer);
    secondHidden.weight = new LayerMatrix(4, 1);
    secondHidden.product = new LayerMatrix(4, 4, 0);
    secondHidden.inputOrActivation = new LayerMatrix(4, 4, 0);
    secondHidden.weightChange = new LayerMatrix(4, 1, 0);
    secondHidden.delta = new LayerMatrix(4, 4, 0);
    layersVector.push_back(std::move(secondHidden));

    Layer outLayer(LayerType::OutputLayer);
    outLayer.product = new LayerMatrix(4, 1, 0);
    outLayer.inputOrActivation = new LayerMatrix(4, 1, 0);
    outLayer.delta = new LayerMatrix(4, 1, 0);

    layersVector.push_back(std::move(outLayer));
}

void NeuralNetwork::forward()
{
    for (unsigned i = 0; i < layersVector.size() - 1; i++)
    {
        *layersVector[i + 1].product =
                *layersVector[i].inputOrActivation *
                *layersVector[i].weight;

        layersVector[i + 1].inputOrActivation->matrix = apply_sigmoid(
                layersVector[i + 1].product->matrix
                );
    }
}

void NeuralNetwork::backPropagate()
{
    for (unsigned i = layersVector.size() - 1; i > 0; i--)
    {
        switch (layersVector[i].getLayerType())
        {
            case LayerType::OutputLayer:

                layersVector[i].delta->matrix =
                        (expectation.matrix - layersVector[i].inputOrActivation->matrix) *
                        apply_sigmoid_d(layersVector[i].product->matrix);

                *layersVector[i - 1].weightChange =
                        layersVector[i - 1].inputOrActivation->transpose() *
                        *layersVector[i].delta;
                break;

            case LayerType::HiddenLayer:

                *layersVector[i].delta =
                        *layersVector[i].weight *
                        layersVector[i + 1].delta->transpose();

                layersVector[i].delta->matrix = layersVector[i].delta->matrix *
                        apply_sigmoid_d(layersVector[i].product->matrix);

                *layersVector[i - 1].weightChange =
                        layersVector[i - 1].inputOrActivation->transpose() *
                        *layersVector[i].delta;

                break;
        }
    }

    for (Layer & layer : layersVector)
    {
        if (layer.getLayerType() != LayerType::OutputLayer)
        {
            layer.weight->matrix = layer.weight->matrix + (scalar * layer.weightChange->matrix);
        }
    }
}

double NeuralNetwork::costFunction()
{           
    LayerMatrix * last = (*--layersVector.end()).inputOrActivation;// layersVector[layersVector.size() - 1].inputOrActivation;
    LayerMatrix cost(last->rows, last->columns, expectation.matrix - last->matrix);
    double costOverall = 0;
    std::for_each(cost.matrix.begin(), cost.matrix.end(), [&costOverall](double & one)->void {
        costOverall += std::fabs(one);
    });
    cout<< "\ncostOverall: " << costOverall << "\n";
    printMatrix(&cost);
}

void NeuralNetwork::execute()
{
    firstTest();
    runLoop();
    costFunction();
}

void NeuralNetwork::runLoop()
{
    unsigned buffer = 100000;

    while (buffer-- > 0)
    {
        forward();
        backPropagate();
    }

    for (vector<Layer>::iterator i = layersVector.begin(); i != layersVector.end(); i++)
    {
        cout << "number: " << ++buffer << "\n";
        cout << (*i);
    }
}

void NeuralNetwork::printMatrix(const LayerMatrix * const v)
{
    for (unsigned i = 0; i < v->rows; i++)
    {
        for (unsigned y = 0; y < v->columns; y++)
        {
            std::cout << std::setw(15) << v->matrix[i * v->columns + y];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void NeuralNetwork::printVector(const vector<double> & v)
{
    for (vector<double>::const_iterator it = v.begin(); it != v.end(); it++)
        std::cout << *it << "  ";
    std::cout << "\n";
}

//Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).

double NeuralNetwork::sigmoid(double value)
{
    return 1 / (1 + exp(-value));
}

//Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)), 

double NeuralNetwork::sigmoid_d(double value)
{
    return sigmoid(value) * (1 - sigmoid(value));
}

vector<double> NeuralNetwork::apply_sigmoid(const vector<double> & matrixOne)
{
    vector<double> matrixResult(matrixOne.size(), 0);

    for (unsigned i = 0; i < matrixOne.size(); i++)
    {
        matrixResult[i] = sigmoid(matrixOne[i]);
    }

    return matrixResult;
}

vector<double> NeuralNetwork::apply_sigmoid_d(const vector<double> & matrixOne)
{
    vector<double> matrixResult(matrixOne.size(), 0);

    for (unsigned i = 0; i < matrixOne.size(); i++)
    {
        matrixResult[i] = sigmoid_d(matrixOne[i]);
    }

    return matrixResult;
}

vector<double> operator+(const vector<double> & matrixOne, const vector<double> & matrixTwo)
{
    vector<double> matrixResult(matrixOne.size(), 0);

    for (unsigned i = 0; i < matrixOne.size(); i++)
    {
        matrixResult[i] = matrixOne[i] + matrixTwo[i];
    }

    return matrixResult;
}

vector<double> operator-(const vector<double> & matrixOne, const vector<double> & matrixTwo)
{
    vector<double> matrixResult(matrixOne.size(), 0);

    for (unsigned i = 0; i < matrixOne.size(); i++)
    {
        matrixResult[i] = matrixOne[i] - matrixTwo[i];
    }

    return matrixResult;
}

vector<double> operator*(const vector<double> & matrixOne, const vector<double> & matrixTwo)
{
    vector<double> matrixResult(matrixOne.size(), 0);

    for (unsigned i = 0; i < matrixOne.size(); i++)
    {
        matrixResult[i] = matrixOne[i] * matrixTwo[i];
    }

    return matrixResult;
}

vector<double> operator*(const double & scalar, const vector<double> & matrixTwo)
{
    vector<double> matrixResult(matrixTwo.size(), 0);

    for (unsigned i = 0; i < matrixTwo.size(); i++)
    {
        matrixResult[i] = matrixTwo[i] * scalar;
    }

    return matrixResult;
}
