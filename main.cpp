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
#include <thread>
#include <mutex> 
#include "NeuralNetwork.h"
using namespace std;

std::mutex mtx; 

void executeNeural(NeuralNetwork * n)
{
    n->execute();
    mtx.lock();
    std::cout << "\n ================================ \n";
    n->show();
    std::cout << "\n ================================ \n";
    mtx.unlock();
}

void testNerual()
{
    std::vector<NeuralNetwork*> tasks;
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++)
    {
        NeuralNetwork * test = new NeuralNetwork();

        test->addScalar(4.8);

        test->addExpectation({
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

        test->addInputLayer({
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

        test->oneHiddenLayer();
        
        tasks.push_back(test);
        
 
    }

    for(NeuralNetwork * it : tasks)
    {
        threads.push_back(std::thread(executeNeural,it));
    }
    
    for (auto& th : threads) th.join();
    
    for(NeuralNetwork * it : tasks)
    {
        delete it;
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
