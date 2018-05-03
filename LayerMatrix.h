/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LayerMatrix.h
 * Author: piotr
 *
 * Created on April 3, 2018, 11:05 AM
 */

#ifndef LAYERMATRIX_H
#define LAYERMATRIX_H
#include <vector>
#include <initializer_list>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <chrono>

using std::vector;
using std::initializer_list;

class LayerMatrix
{
protected:
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution;
    static double randomGenerator();
public:
    vector<double> matrix;
    unsigned int rows;
    unsigned int columns;

    LayerMatrix();
    LayerMatrix(const LayerMatrix & one);
    LayerMatrix(LayerMatrix && one);
    LayerMatrix(unsigned int r, unsigned int c);
    LayerMatrix(unsigned int r, unsigned int c, double v);
    LayerMatrix(unsigned int r, unsigned int c, vector<double> v);
    LayerMatrix(unsigned int r, unsigned int c, initializer_list<double> v);
    LayerMatrix & operator=(const LayerMatrix & one);
    LayerMatrix operator*(const LayerMatrix & two);
    LayerMatrix transpose();

};

#endif /* LAYERMATRIX_H */

