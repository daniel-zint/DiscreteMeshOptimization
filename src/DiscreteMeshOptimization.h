/**
 * Copyright (C) 2018 by Daniel Zint and Philipp Guertler
 * This file is part of Discrete Mesh Optimization DMO
 * Some rights reserved. See LICENCE.
 */

#pragma once

// std includes
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

// OpenMesh includes
#include "OpenMeshConfig.h"

// Eigen includes
#include "Eigen/Dense"

// other
#include "Stopwatch.h"
#include "QualityCriteria.h"
#include "DMOConfig.h"


namespace DMO {

	// ##################################################################### //
	// ### CUDA functions in "kernel.cu" ################################### //
	// ##################################################################### //

	void initCuda();
	void discreteMeshOptimization(DMOMesh& mesh, QualityCriterium qualityCriterium = MEAN_RATIO, const float gridScale = 0.5f, int n_iter = 100);


	// ##################################################################### //
	// ### CPU functions in "DiscreteMeshOptimization.cpp" ################# //
	// ##################################################################### //
	
	void discreteMeshOptimizationCPU(PolyMesh& mesh, QualityCriterium qualityCriterium = MEAN_RATIO, const float gridScale = 0.5f, int n_iter = 100);

	// ##################################################################### //
	// ### Metric functions ################################################ //

	float metricMeanRatioTri(TriMesh& mesh, const TriMesh::FaceHandle& fh);
	float metricMeanRatioQuad(PolyMesh& mesh, const PolyMesh::FaceHandle& fh);
	float metricMeanRatioQuad(Eigen::Vector2f points[4]);

	// ##################################################################### //
	// ### Print Quality functions ######################################### //
	
	void printQuality(TriMesh& mesh, std::ofstream& ofs);
	void displayQuality(TriMesh& mesh, int n_columns);
	void displayQuality(PolyMesh& mesh, int n_columns);
}