/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   NeuralNetwork.h
 * Author: piotr
 *
 * Created on March 18, 2018, 11:53 AM
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <initializer_list>
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <chrono>

#include "LayerType.h"
#include "Layer.h"
#include "LayerMatrix.h"

using std::vector;
using std::cout;
using std::endl;
using namespace std::chrono;

vector<double> operator+(const vector<double> & matrixOne, const vector<double> & matrixTwo);
vector<double> operator-(const vector<double> & matrixOne, const vector<double> & matrixTwo);
vector<double> operator*(const vector<double> & matrixOne, const vector<double> & matrixTwo);
vector<double> operator*(const double & scalar, const vector<double> & matrixTwo);

class NeuralNetwork
{
protected:
    double scalar;
    double costOverall;
    vector<double> costVector;
    LayerMatrix expectation;
    vector<Layer> layersVector;
public:
    NeuralNetwork();
    void firstTest();
    void secondTest();
    void addExpectation(initializer_list<double> v, unsigned rows, unsigned columns);
    void addScalar(double scalar);
    void addInputLayer(initializer_list<double> v, unsigned rows, unsigned columns);
    void oneHiddenLayer();
    void addHiddenLayers(initializer_list<double> v);
    void execute();
    void show();
protected:
    void runLoop();
    void forward();
    void backPropagate();
    void weightUpdate();
    double costFunction();
    double sigmoid(double value);
    double sigmoid_d(double value);
    vector<double> apply_sigmoid(const vector<double> & matrixOne);
    vector<double> apply_sigmoid_d(const vector<double> & matrixOne);
    void printMatrix(const LayerMatrix * const v);
    void printVector(const vector<double> & v);
};

#endif /* NEURALNETWORK_H */

