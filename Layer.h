/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Layer.h
 * Author: piotr
 *
 * Created on April 3, 2018, 10:18 AM
 */

#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <iostream>
#include <iomanip>
#include "LayerType.h"
#include "LayerMatrix.h"

using std::ostream;

class Layer
{
public:
    LayerMatrix * inputOrActivation = nullptr;
    LayerMatrix * product = nullptr;
    LayerMatrix * weight = nullptr;
    LayerMatrix * weightChange = nullptr;
    LayerMatrix * delta = nullptr;
protected:
    LayerType layerType;
public:
    Layer(LayerType);
    Layer(const Layer & one);
    Layer(Layer && one);
    ~Layer();
    LayerType getLayerType();
    Layer & operator=(const Layer & one);
    ostream & printOneLayerMatrix(ostream & out, const LayerMatrix * const one, const char * type);
    friend ostream & operator<<(ostream & out, Layer & one);
};


#endif /* LAYER_H */

