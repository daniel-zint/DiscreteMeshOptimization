#include "stopwatch.h"

#include <iostream>
#include <sstream>
#include <ctime>

Stopwatch::Stopwatch()
{
	tStart = 0.;
	tStop = 0.;
	dimFactor = 1;
	dimFactorStr = "ms";
}

Stopwatch::Stopwatch(const std::string &dim)
{
	dimFactorStr = dim;

	if (dim == "s")
	{
		dimFactor = 0.001;
	}
	else if (dim == "ms")
	{
		dimFactor = 1;
	}
	else
	{
		dimFactor = 1;
		dimFactorStr = "ms";
	}
}

unsigned int Stopwatch::start()
{
	tStart = clock();

	return tStart;
}

unsigned int Stopwatch::stop()
{
	tStop = clock();

	return tStop;
}


unsigned int Stopwatch::runtime()
{
	// in seconds
	return (tStop - tStart) * dimFactor;
}

std::string Stopwatch::runtimeStr()
{
	// in seconds
	std::ostringstream strs;
	strs << runtime();
	return strs.str() + " " + dimFactorStr;
}