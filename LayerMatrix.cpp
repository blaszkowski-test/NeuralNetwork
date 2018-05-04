/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "LayerMatrix.h"

LayerMatrix::LayerMatrix() : rows(0), columns(0)
{
}

LayerMatrix::LayerMatrix(unsigned int r, unsigned int c, double v) :
rows(r), columns(c)
{
    matrix.resize(rows*columns, v);
}

LayerMatrix::LayerMatrix(unsigned int r, unsigned int c) :
rows(r), columns(c)
{
    matrix.resize(rows*columns, 0);
    for (double & one : matrix)
    {
        one = LayerMatrix::randomGenerator();
    }
}

LayerMatrix::LayerMatrix(unsigned int r, unsigned int c, vector<double> v) :
rows(r), columns(c)
{
    matrix.assign(v.begin(), v.end());
}

LayerMatrix::LayerMatrix(unsigned int r, unsigned int c, initializer_list<double> v) :
rows(r), columns(c)
{
    matrix.assign(v.begin(), v.end());
}

LayerMatrix::LayerMatrix(const LayerMatrix & one) :
rows(one.rows), columns(one.columns), matrix(one.matrix)
{
}

LayerMatrix::LayerMatrix(LayerMatrix && one) :
rows(one.rows), columns(one.columns), matrix(one.matrix)
{
    one.rows = 0;
    one.columns = 0;
    one.matrix.clear();
}

LayerMatrix & LayerMatrix::operator=(const LayerMatrix & one)
{
    if (&one == this)
        return *this;

    rows = one.rows;
    columns = one.columns;
    matrix.assign(one.matrix.begin(), one.matrix.end());

    return *this;
}

LayerMatrix LayerMatrix::operator*(const LayerMatrix & two)
{
    LayerMatrix result(this->rows, two.columns, 0);

    for (unsigned int rowCount = 0; rowCount < this->rows; rowCount++)
    {
        for (unsigned int columnCount = 0; columnCount < two.columns; columnCount++)
        {
            for (unsigned int commonCount = 0; commonCount < this->columns; commonCount++)
            {
                result.matrix[rowCount * two.columns + columnCount] +=
                        this->matrix[rowCount * this->columns + commonCount] *
                        two.matrix[commonCount * two.columns + columnCount];
            }
        }
    }
    return result;
}

LayerMatrix LayerMatrix::transpose()
{
    LayerMatrix result(this->columns, this->rows, 0);

    for (unsigned rowCounter = 0; rowCounter < this->rows; rowCounter++)
    {
        for (unsigned columnsCounter = 0; columnsCounter < this->columns; columnsCounter++)
        {
            result.matrix[columnsCounter * this->rows + rowCounter] =
                    this->matrix[rowCounter * this->columns + columnsCounter];
        }
    }

    return result;
}

std::default_random_engine LayerMatrix::generator(std::chrono::system_clock::now().time_since_epoch().count());

std::uniform_real_distribution<double> LayerMatrix::distribution(0.0, 1.0);

double LayerMatrix::randomGenerator()
{
    return distribution(generator);
}