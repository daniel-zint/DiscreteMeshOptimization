#pragma once
#include <string>

class Stopwatch
{
	double tStart;
	double tStop;
	double dimFactor;
	std::string dimFactorStr;

public:
	Stopwatch();
	Stopwatch(const std::string &dim);

	unsigned int start();
	unsigned int stop();
	unsigned int runtime();
	std::string runtimeStr();
};

