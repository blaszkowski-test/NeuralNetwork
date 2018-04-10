/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: piotr
 *
 * Created on March 18, 2018, 11:52 AM
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.h"
using namespace std;

void testVector()
{
    const int size = 10;

    vector<int*> firstTable;
    vector<int*> secondTable;

    for (int i = 0; i < size; i++)
    {
        firstTable.push_back(new int(i));
    }

    for (int i = 0; i < size; i++)
    {
        secondTable.push_back(firstTable[i]);
    }

    for (int i = 0; i < size; i++)
    {
        cout << *secondTable[i] << "\n";
    }



    for (int i = 0; i < size; i++)
    {
        delete firstTable[i];
    }
}

void testNerual()
{
    srand (time(NULL));
    NeuralNetwork test;
    test.execute();
}

/*
 * 
 */
int main(int argc, char** argv)
{
    testNerual();
    return 0;
}

