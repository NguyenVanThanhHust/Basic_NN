#include <iostream>
#include <sstream>
#include <fstream>
#include <random>

#include "linear.h"

using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
    if (argc!=2)
    {
        cout<<"Usage: ./linear ../data.txt";
        return -1;
    }
    
    int trainSize = 2000, testSize=400;
    std::ifstream infile("../data.txt");
    float ele1, ele2;
    int label;
    while (infile >> ele1 >> ele2 >> label)
    {
        cout<<elel<<" "<<ele2<<" "<<label<<endl;
    }
    
}