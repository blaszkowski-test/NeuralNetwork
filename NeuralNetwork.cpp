/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <algorithm>
#include <cmath>

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() :
scalar(0),
costOverall(0),
lastCost(0),
bestCost(std::numeric_limits<double>::max())
{
}

void NeuralNetwork::addExpectation(initializer_list<double> v, unsigned rows, unsigned columns)
{
    expectation.rows = rows;
    expectation.columns = columns;
    expectation.matrix.assign(v.begin(), v.end());
}

void NeuralNetwork::addScalar(double scalar)
{
    this->scalar = scalar;
}

void NeuralNetwork::addInputLayer(initializer_list<double> v, unsigned rows, unsigned columns)
{
    Layer inLayer(LayerType::InputLayer);
    inLayer.inputOrActivation = new LayerMatrix(rows, columns, v);
    inLayer.weight = new LayerMatrix(columns, rows);
    inLayer.weightChange = new LayerMatrix(columns, rows, 0);
    layersVector.push_back(std::move(inLayer));
}

void NeuralNetwork::oneHiddenLayer()
{
    Layer * layerPointer = &(*--layersVector.end());
    unsigned rows = layerPointer->inputOrActivation->rows;
    unsigned columns = layerPointer->weight->columns;

    Layer hidden(LayerType::HiddenLayer);
    hidden.product = new LayerMatrix(rows, columns, 0);
    hidden.inputOrActivation = new LayerMatrix(rows, columns, 0);
    hidden.delta = new LayerMatrix(rows, columns, 0);
    hidden.weightChange = new LayerMatrix(columns, expectation.columns, 0);
    hidden.weight = new LayerMatrix(columns, expectation.columns);
    layersVector.push_back(std::move(hidden));

    Layer outLayer(LayerType::OutputLayer);
    outLayer.product = new LayerMatrix(expectation.rows, expectation.columns, 0);
    outLayer.inputOrActivation = new LayerMatrix(expectation.rows, expectation.columns, 0);
    outLayer.delta = new LayerMatrix(expectation.rows, expectation.columns, 0);
    layersVector.push_back(std::move(outLayer));
}

void NeuralNetwork::addHiddenLayers(initializer_list<double> v)
{
    vector<double> hiddenNodes(v.begin(), v.end());
    Layer * layerPointer = nullptr;
    unsigned rows = 0;
    unsigned columns = 0;
    for (int i = -1; i < (int) hiddenNodes.size(); i++)
    {
        layerPointer = &(*--layersVector.end());
        rows = layerPointer->inputOrActivation->rows;
        columns = layerPointer->weight->columns;

        Layer hidden(LayerType::HiddenLayer);
        hidden.product = new LayerMatrix(rows, columns, 0);
        hidden.inputOrActivation = new LayerMatrix(rows, columns, 0);
        hidden.delta = new LayerMatrix(rows, columns, 0);

        if (i == hiddenNodes.size() - 1)
        {
            hidden.weightChange = new LayerMatrix(hidden.inputOrActivation->columns, expectation.columns, 0);
            hidden.weight = new LayerMatrix(hidden.inputOrActivation->columns, expectation.columns);
        }
        else
        {
            hidden.weightChange = new LayerMatrix(hidden.inputOrActivation->columns, hiddenNodes[i + 1], 0);
            hidden.weight = new LayerMatrix(hidden.inputOrActivation->columns, hiddenNodes[i + 1]);
        }

        layersVector.push_back(std::move(hidden));
    }

    Layer outLayer(LayerType::OutputLayer);
    outLayer.product = new LayerMatrix(expectation.rows, expectation.columns, 0);
    outLayer.inputOrActivation = new LayerMatrix(expectation.rows, expectation.columns, 0);
    outLayer.delta = new LayerMatrix(expectation.rows, expectation.columns, 0);
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
}

void NeuralNetwork::weightUpdate()
{
    for (Layer & layer : layersVector)
    {
        if (layer.getLayerType() != LayerType::OutputLayer)
        {
            layer.weight->matrix = layer.weight->matrix + (scalar * layer.weightChange->matrix);
        }
    }
}

void NeuralNetwork::costFunction()
{
    lastCost = costOverall;
    costOverall = 0;
    costVector = expectation.matrix - (*--layersVector.end()).inputOrActivation->matrix;
    std::for_each(costVector.begin(), costVector.end(), [this](double & one)->void {
        this->costOverall += std::fabs(one);
    });
    if (bestCost > costOverall)
    {
        bestCost = costOverall;
        copyBestResult();
    }
}

void NeuralNetwork::createResultStructure()
{
    for (Layer & layer : layersVector)
    {
        if (layer.getLayerType() != LayerType::OutputLayer)
        {
            bestResult.push_back(*layer.weight);
        } else
        {
            bestResult.push_back(*layer.inputOrActivation);
        }
    }
}

void NeuralNetwork::copyBestResult()
{
    for (unsigned i = 0; i < layersVector.size(); i++)
    {
        if (layersVector[i].getLayerType() != LayerType::OutputLayer)
        {
            bestResult[i] = *layersVector[i].weight;
        }
        else
        {
            bestResult[i] = *layersVector[i].inputOrActivation;
        }
    }
}

void NeuralNetwork::show()
{
    double buffer = 0;
    for (vector<Layer>::iterator i = layersVector.begin(); i != layersVector.end(); i++)
    {
        cout << "number: " << ++buffer << "\n";
        cout << (*i);
    }
}

void NeuralNetwork::check()
{
    forward();
    show();
}

void NeuralNetwork::execute()
{
    createResultStructure();
    runLoop();
}

void NeuralNetwork::runLoop()
{
    steady_clock::time_point t1 = steady_clock::now();

    while (true)
    {
        forward();
        backPropagate();
        weightUpdate();
        costFunction();

        if (std::abs(lastCost - costOverall) < 1e-09 ||
                duration_cast<duration<double>>(steady_clock::now() - t1).count() > 60)
        {
            break;
        }
    }

    //show();
    cout << "\n\nCost: " << costOverall << "\n\n";
    cout << "\n\nBest Cost: " << bestCost << "\n\n";

    for (LayerMatrix & one : bestResult)
    {
        printMatrix(&one);
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
