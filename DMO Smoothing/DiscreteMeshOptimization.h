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


namespace DMO {

	// ##################################################################### //
	// ### CUDA functions ################################################## //
	// ##################################################################### //

	void initCuda();
	void discreteMeshOptimization(TriMesh& mesh, QualityCriterium qualityCriterium = MEAN_RATIO, const float gridScale = 0.5f, int n_iter = 100);
	void discreteMeshOptimization(PolyMesh& mesh, QualityCriterium qualityCriterium = MEAN_RATIO, const float gridScale = 0.5f, int n_iter = 100);


	// ##################################################################### //
	// ### CPU functions ################################################### //
	// ##################################################################### //
	
	void discreteMeshOptimizationCPU(PolyMesh& mesh, const int qualityCriterium = MEAN_RATIO, const float gridScale = 0.5f, int n_iter = 100);


	// ##################################################################### //
	// ### Metric functions ################################################ //
	// ##################################################################### //

	float metricMeanRatioTri(TriMesh& mesh, const TriMesh::FaceHandle& fh);
	float metricMeanRatioQuad(PolyMesh& mesh, const PolyMesh::FaceHandle& fh);
	float metricMeanRatioQuad(Eigen::Vector2f points[4]);


	// ##################################################################### //
	// ### Print Quality functions ######################################### //
	// ##################################################################### //
	
	void printQuality(TriMesh& mesh, std::ofstream& ofs);
	void displayQuality(TriMesh& mesh, int n_columns);
	void displayQuality(PolyMesh& mesh, int n_columns);

}