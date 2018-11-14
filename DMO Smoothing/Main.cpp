/**
 * Copyright (C) 2018 by Daniel Zint and Philipp Guertler
 * This file is part of Discrete Mesh Optimization DMO
 * Some rights reserved. See LICENCE.
 */

#include "DiscreteMeshOptimization.h"

int main() {

	DMO::initCuda();
	Stopwatch watch;

	// Enter path to your Mesh here
	std::string fileName = "../Meshes/example.obj";

	TriMesh mesh;
	// read mesh from file:
	if (!OpenMesh::IO::read_mesh(mesh, fileName)) {
		std::cerr << "Error reading file \"" << fileName << "\"." << std::endl;
		return -1;
	}

	std::cout << "Mesh loaded with " << mesh.n_vertices() << " Vertices." << std::endl;

	DMO::displayQuality(mesh, 10);

	std::cout << "### running DMO ###" << std::endl;
	watch.start();

	// --- the actual calculation --- //
	DMO::discreteMeshOptimization(mesh, DMO::MEAN_RATIO, 0.5, 4);
	// ------------------------------ //

	watch.stop();
	std::cout << "### DMO finished ###" << std::endl;
	std::cout << "Runtime: " << watch.runtimeStr() << std::endl;

	DMO::displayQuality(mesh, 10);

	// save smoothened mesh to file
	std::string outputFileName = "../Meshes/output.obj";
	std::cout << "Saving Mesh to " << outputFileName << "." << std::endl;

	if (!OpenMesh::IO::write_mesh(mesh, outputFileName)) {
		std::cerr << "Error writing mesh to file \"" << outputFileName << "\"" << std::endl;
		return -1;
	}

	std::cout << "Saving complete." << std::endl;

	// wait for userinput to exit program
	std::cout << "Press Enter to exit..." << std::endl;
	std::cin.get();
}