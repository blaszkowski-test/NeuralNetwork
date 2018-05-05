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
#include <thread>
#include <mutex> 
#include "NeuralNetwork.h"
using namespace std;

void executeNeural(NeuralNetwork & n)
{
    n.execute();
}

void testNerual()
{
    std::vector<NeuralNetwork> tasks;
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++)
    {
        tasks.push_back(NeuralNetwork());

        NeuralNetwork & bufferRef = (*--tasks.end());

        bufferRef.addScalar(4.8);

        bufferRef.addExpectation({
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
        }, 10, 4);

        bufferRef.addInputLayer({
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
        }, 10, 10);

        bufferRef.oneHiddenLayer();
    }

    for (NeuralNetwork & it : tasks)
    {
        threads.push_back(std::thread(executeNeural, std::ref(it)));
    }

    for (auto& th : threads) th.join();

    std::sort(tasks.begin(), tasks.end());
    for (auto& t : tasks)
    {
        t.show();
    }
}

/*
 * 
 */
int main(int argc, char** argv)
{
    testNerual();
    return 0;
}
