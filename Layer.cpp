/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "Layer.h"

Layer::Layer(LayerType l) : layerType(l)
{
}

Layer::Layer(const Layer & one) : layerType(one.layerType)
{
    //std::cout << (unsigned long) this << " Layer COPY constructor" << "\n";

    if (one.inputOrActivation != nullptr)
        this->inputOrActivation = new LayerMatrix(*(one.inputOrActivation));
    if (one.product != nullptr)
        this->product = new LayerMatrix(*(one.product));
    if (one.weight != nullptr)
        this->weight = new LayerMatrix(*(one.weight));
    if (one.weightChange != nullptr)
        this->weightChange = new LayerMatrix(*(one.weightChange));
    if (one.delta != nullptr)
        this->delta = new LayerMatrix(*(one.delta));
}

Layer::Layer(Layer && one) : layerType(one.layerType)
{
    //std::cout << (unsigned long) this << " Layer MOVE constructor" << "\n";

    this->inputOrActivation = one.inputOrActivation;
    this->product = one.product;
    this->weight = one.weight;
    this->weightChange = one.weightChange;
    this->delta = one.delta;

    one.inputOrActivation = nullptr;
    one.product = nullptr;
    one.weight = nullptr;
    one.weightChange = nullptr;
    one.delta = nullptr;
}

LayerType Layer::getLayerType()
{
    return this->layerType;
}

Layer::~Layer()
{
    //std::cout << (unsigned long) this<< " Layer Destructor" << "\n";
    if (inputOrActivation != nullptr)
        delete inputOrActivation;
    if (product != nullptr)
        delete product;
    if (weight != nullptr)
        delete weight;
    if (weightChange != nullptr)
        delete weightChange;
    if (delta != nullptr)
        delete delta;
}

Layer & Layer::operator=(const Layer & one)
{
    if (&one == this)
        return *this;

    this->layerType = one.layerType;

    if (one.inputOrActivation != nullptr)
        this->inputOrActivation = new LayerMatrix(*(one.inputOrActivation));
    if (one.product != nullptr)
        this->product = new LayerMatrix(*(one.product));
    if (one.weight != nullptr)
        this->weight = new LayerMatrix(*(one.weight));
    if (one.weightChange != nullptr)
        this->weightChange = new LayerMatrix(*(one.weightChange));
    if (one.delta != nullptr)
        this->delta = new LayerMatrix(*(one.delta));

    return *this;
}

ostream & Layer::printOneLayerMatrix(ostream & out, const LayerMatrix * const one, const char * type)
{
    if (one == nullptr)
        return out;

    out << type << ":\n";

    for (unsigned i = 0; i < one->rows; i++)
    {
        for (unsigned y = 0; y < one->columns; y++)
        {
            out << std::setw(5) << one->matrix[i * one->columns + y]
                    << std::setw(2) << " , ";
        }
        out << "\n";
    }
    out << "\n";

    return out;
}

ostream & operator<<(ostream& out, Layer& one)
{
    one.printOneLayerMatrix(out, one.inputOrActivation, "inputOrActivation");
    one.printOneLayerMatrix(out, one.product, "product");
    one.printOneLayerMatrix(out, one.weight, "weight");
    one.printOneLayerMatrix(out, one.weightChange, "weightChange");
    one.printOneLayerMatrix(out, one.delta, " delta");
    return out;
}