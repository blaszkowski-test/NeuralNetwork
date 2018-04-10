/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
}

void NeuralNetwork::firstTest()
{
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
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0

    });
    inLayer.weight = new LayerMatrix(10, 10);
    inLayer.weightChange = new LayerMatrix(10, 10, 0);
    inLayer.delta = new LayerMatrix(10, 10, 0);
    layersVector.push_back(std::move(inLayer));

    Layer firstHidden(LayerType::HiddenLayer);
    firstHidden.weight = new LayerMatrix(10, 10);
    firstHidden.product = new LayerMatrix(10, 10, 0);
    firstHidden.inputOrActivation = new LayerMatrix(10, 10, 0);
    firstHidden.weightChange = new LayerMatrix(10, 10, 0);
    firstHidden.delta = new LayerMatrix(10, 10, 0);
    layersVector.push_back(std::move(firstHidden));

    Layer secondHidden(LayerType::HiddenLayer);
    secondHidden.weight = new LayerMatrix(10, 4);
    secondHidden.product = new LayerMatrix(10, 10, 0);
    secondHidden.inputOrActivation = new LayerMatrix(10, 10, 0);
    secondHidden.weightChange = new LayerMatrix(10, 4, 0);
    secondHidden.delta = new LayerMatrix(10, 10, 0);
    layersVector.push_back(std::move(secondHidden));

    Layer outLayer(LayerType::OutputLayer);
    outLayer.product = new LayerMatrix(10, 4, 0);
    outLayer.inputOrActivation = new LayerMatrix(10, 4, 0);
    outLayer.delta = new LayerMatrix(10, 4, 0);

    layersVector.push_back(std::move(outLayer));
}

void NeuralNetwork::secondTest()
{
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

void NeuralNetwork::execute()
{
    secondTest();
    runLoop();
}

void NeuralNetwork::runLoop()
{
    unsigned buffer = 10000;

    while (buffer-- > 0)
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
                layer.weight->matrix = layer.weight->matrix + layer.weightChange->matrix;
            }
        }
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

vector<double> NeuralNetwork::matrixMultiply(const vector<double> & firstMatrix,
        const vector<double> & secondMatrix,
        const unsigned int firstColumnCount,
        const unsigned int firstRowCount,
        const unsigned int secondColumnCount,
        const unsigned int secondRowCount)
{
    vector<double> result(firstRowCount * secondColumnCount, 0);

    for (unsigned int rowCount = 0; rowCount < firstRowCount; rowCount++)
    {
        for (unsigned int columnCount = 0; columnCount < secondColumnCount; columnCount++)
        {
            for (unsigned int commonCount = 0; commonCount < firstColumnCount; commonCount++)
            {
                result[rowCount * secondColumnCount + columnCount] +=
                        firstMatrix[rowCount * firstColumnCount + commonCount] *
                        secondMatrix[commonCount * secondColumnCount + columnCount];
            }
        }
    }
    return result;
}

vector<double> NeuralNetwork::transpose(const LayerMatrix * const v)
{
    vector<double> matrixResult(v->matrix.size(), 0);

    for (unsigned rowCounter = 0; rowCounter < v->rows; rowCounter++)
    {
        for (unsigned columnsCounter = 0; columnsCounter < v->columns; columnsCounter++)
        {
            matrixResult[columnsCounter * v->rows + rowCounter] =
                    v->matrix[rowCounter * v->columns + columnsCounter];
        }
    }

    return matrixResult;
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
