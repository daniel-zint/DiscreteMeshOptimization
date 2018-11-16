/** 
 * Copyright (C) 2018 by Daniel Zint and Philipp Guertler
 * This file is part of Discrete Mesh Optimization DMO
 * Some rights reserved. See LICENCE.
 *
 * The CUDA implementation of Discrete Mesh Optimization.
 */

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

// C includes
#include <stdio.h>
#include <cfloat>

// OpenMesh includes
#include "OpenMeshConfig.h"

// other includes
#include "Stopwatch.h"
#include "QualityCriteria.h"
#include "DMOConfig.h"

namespace DMO {

	// ######################################################################### //
	// ### Misc Functions ###################################################### //
	// ######################################################################### //

	// Error output
	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s. Line %d\n", cudaGetErrorString(code), line);
			if (abort) exit(code);
		}
	}

	// Initialize CUDA. It is not necessary but recommended as initialization takes about 200 ms. Call this function early in your code.
	void initCuda() {
		cudaSetDevice(0);
		cudaFree(0);
	}

	typedef union {
		float floats[2];                 // floats[0] = lowest
		int ints[2];                     // ints[1] = lowIdx
		unsigned long long int ulong;    // for atomic update
	} my_atomics;

	__device__ unsigned long long int my_atomicArgMax(unsigned long long int* address, float val1, int val2)
	{
		my_atomics loc, loctest;
		loc.floats[0] = val1;
		loc.ints[1] = val2;
		loctest.ulong = *address;
		while (loctest.floats[0] < val1)
			loctest.ulong = atomicCAS(address, loctest.ulong, loc.ulong);
		return loctest.ulong;
	}
	
	// helper: calculates angle between [p1-p2] and [p3-p2]
	__host__ __device__ __forceinline__ float calcAngle(const float p1[2], const float p2[2], const float p3[2]) {
		const float dot = (p1[0] - p2[0]) * (p3[0] - p2[0]) + (p1[1] - p2[1]) * (p3[1] - p2[1]);
		const float det = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0]);
		return fabsf(atan2f(det, dot));
	}

	/*
	Holds information of the vertex-id, the number of neighbors, and their location in the one-ring vector.
	*/
	struct Vertex {
		int oneRingID;
		int n_oneRing;
		int id;				// own vertex id
	};


	// ######################################################################### //
	// ##################### Quality Metrics per Element ####################### //
	//                                                                           //
	//    These Metrics take a single Triangle or a Single Quad and calculate    //
	//    its quality.                                                           //
	// ######################################################################### //


	// ######################################################################### //
	// ### Triangle Metrics: ################################################### //

	// ### MEAN_RATIO Metric on Triangles ###################################### //
	__host__ __device__ __forceinline__ float metricMeanRatioTri(const float p[3][2]) {

		float e[3][2];
		float e_length_squared[3];

		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i][0] = p[j][0] - p[i][0];
			e[i][1] = p[j][1] - p[i][1];

			e_length_squared[i] = e[i][0] * e[i][0] + e[i][1] * e[i][1];
		}

		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2];
		float area = e[0][0] * e[1][1] - e[0][1] * e[1][0];

		return 2.f * sqrt(3.f) * area / l;
	}

	// ### RIGHT_ANGLE Metric on Triangles ##################################### //
	__host__ __device__ __forceinline__ float metricRectangularityTri(const float p[3][2]) {
		float e[3][2];
		float e_length_squared[3];
		float e_length[3];

		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i][0] = p[j][0] - p[i][0];
			e[i][1] = p[j][1] - p[i][1];

			e_length_squared[i] = e[i][0] * e[i][0] + e[i][1] * e[i][1];
			e_length[i] = sqrtf(e_length_squared[i]);
		}

		// angles
		float a[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i - 1 + 3) % 3;
			float dot = e[i][0] * e[j][0] + e[i][1] * e[j][1];
			float det = e[i][0] * e[j][1] - e[i][1] * e[j][0];
			a[i] = CUDART_PI - fabsf(atan2f(det, dot));
		}

		float q_tri = -FLT_MAX;

		for (int i = 0; i < 3; ++i) {
			int j = (i - 1 + 3) % 3;
			float qa = 1.f - fabsf(CUDART_PIO2_F - a[i]) / CUDART_PIO2_F;
			float qc = 1.f - fabsf(e_length[i] - e_length[j]) / fmaxf(e_length[i], e_length[j]);

			q_tri = fmaxf(q_tri, qa * qc);
		}

		return q_tri;
	}

	// ### RADIUS_RATIO Metric for Triangles ################################### //
	__host__ __device__ __forceinline__ float metricRadiusRatioTri(const float p[3][2]) {

		float e[3][2];
		float e_length[3];

		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i][0] = p[j][0] - p[i][0];
			e[i][1] = p[j][1] - p[i][1];

			e_length[i] = sqrtf(e[i][0] * e[i][0] + e[i][1] * e[i][1]);
		}

		float area = 0.5 * (e[0][0] * e[1][1] - e[0][1] * e[1][0]);

		float s = 0.5f * (e_length[0] + e_length[1] + e_length[2]);

		float ri = area / s;
		float ro = 0.25f * e_length[0] * e_length[1] * e_length[2] / area;

		float criterion = (float)(2.f * ri / ro);

		if (criterion > 1) {	// might happen due to numerical errors
			return 1;
		}

		return criterion;
	}

	// ### MAX_ANGLE Metric for Triangles ###################################### //
	__host__ __device__ __forceinline__ float metricMaxAngleTri(const float p[3][2]) {

		float e[3][2];

		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i][0] = p[j][0] - p[i][0];
			e[i][1] = p[j][1] - p[i][1];
		}

		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);

		float area = e[0][0] * e[1][1] - e[0][1] * e[1][0];

		float max_angle = fmaxf(a, b);
		max_angle = fmaxf(max_angle, c);

		float q = 1 - (3.f * max_angle - CUDART_PI) / (2.f * CUDART_PI);

		if (area < 0)
			return -q;
		else
			return q;
	}

	// ### MIN_ANGLE Metric for Triangles ###################################### //
	__host__ __device__ __forceinline__ float metricMinAngleTri(const float p[3][2]) {

		float e[3][2];

		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i][0] = p[j][0] - p[i][0];
			e[i][1] = p[j][1] - p[i][1];
		}

		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);

		float min_angle = fminf(a, b);
		min_angle = fminf(min_angle, c);

		float area = e[0][0] * e[1][1] - e[0][1] * e[1][0];

		float q = 3.f * min_angle / CUDART_PI;

		// if triangle is flipped, make value negative
		if (area < 0)
			return -q;
		else
			return q;
	}


	// ######################################################################### //
	// ### Quad Metrics: ####################################################### //

	// ### MEAN_RATIO Metric on Quads ########################################## //
	__host__ __device__ __forceinline__ float metricMeanRatioQuad(const float p[4][2]) {

		float e[4][2];
		float e_length_squared[4];

		for (int i = 0; i < 4; ++i) {
			int j = (i + 1) % 4;
			e[i][0] = p[j][0] - p[i][0];
			e[i][1] = p[j][1] - p[i][1];

			e_length_squared[i] = e[i][0] * e[i][0] + e[i][1] * e[i][1];
		}

		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2] + e_length_squared[3];
		float area1 = e[0][0] * e[1][1] - e[0][1] * e[1][0];
		float area2 = e[2][0] * e[3][1] - e[2][1] * e[3][0];

		return 2.f * (area1 + area2) / l;
	}

	// ### JACOBIAN Metric on Quads ############################################ //
	__host__ __device__ __forceinline__ float metricScaledJacobianQuad(const float p[4][2]) {

		// get jacobian in every point of the quad
		// take the smallest one
		// scale it with the incident edges

		float dxds_1 = (p[0][0] - p[1][0]);						// dx/ds (t = 1)
		float dxds_2 = (p[3][0] - p[2][0]);						// dx/ds (t = -1)

		float dyds_1 = (p[0][1] - p[1][1]);						// dy/ds (t = 1)
		float dyds_2 = (p[3][1] - p[2][1]);						// dy/ds (t = -1)

		float dxdt_1 = (p[0][0] - p[3][0]);						// dx/dt (s = 1)
		float dxdt_2 = (p[1][0] - p[2][0]);						// dx/dt (s = -1)

		float dydt_1 = (p[0][1] - p[3][1]);						// dy/dt (s = 1)
		float dydt_2 = (p[1][1] - p[2][1]);						// dy/dt (s = -1)

		float j[4];

		j[0] = dxds_1 * dydt_1 - dxdt_1 * dyds_1;		// t = s = 1		// x1
		j[1] = dxds_1 * dydt_2 - dxdt_2 * dyds_1;		// t = 1, s = -1	// x2
		j[2] = dxds_2 * dydt_2 - dxdt_2 * dyds_2;		// t = s = -1		// x3
		j[3] = dxds_2 * dydt_1 - dxdt_1 * dyds_2;		// t = -1, s = 1	// x4

		float jMin = FLT_MAX;
		int ji = 5;
		for (int i = 0; i < 4; ++i) {
			if (j[i] < jMin) {
				jMin = j[i];
				ji = i;
			}
		}

		int jil = (ji + 4 - 1) % 4;
		int jir = (ji + 1) % 4;

		float lEdge = sqrtf((p[jil][0] - p[ji][0]) * (p[jil][0] - p[ji][0]) + (p[jil][1] - p[ji][1]) * (p[jil][1] - p[ji][1]));
		float rEdge = sqrtf((p[jir][0] - p[ji][0]) * (p[jir][0] - p[ji][0]) + (p[jir][1] - p[ji][1]) * (p[jir][1] - p[ji][1]));

		return j[ji] / (lEdge * rEdge);
	}

	// ### MIN_ANGLE Metric for Quads ########################################## //
	__host__ __device__ __forceinline__ float metricMinAngleQuad(const float p[4][2]) {

		float e[4][2];
		float e_length[4];

		for (int i = 0; i < 4; ++i) {
			int j = (i + 1) % 4;
			e[i][0] = p[j][0] - p[i][0];
			e[i][1] = p[j][1] - p[i][1];

			e_length[i] = sqrtf(e[i][0] * e[i][0] + e[i][1] * e[i][1]);
		}

		float a = CUDART_PI - acosf((e[0][0] * e[1][0] + e[0][1] * e[1][1]) / (e_length[0] * e_length[1]));
		float b = CUDART_PI - acosf((e[1][0] * e[2][0] + e[1][1] * e[2][1]) / (e_length[1] * e_length[2]));
		float c = CUDART_PI - acosf((e[2][0] * e[3][0] + e[2][1] * e[3][1]) / (e_length[2] * e_length[3]));
		float d = CUDART_PI - acosf((e[3][0] * e[0][0] + e[3][1] * e[0][1]) / (e_length[3] * e_length[0]));

		float min_angle = fminf(a, b);
		min_angle = fminf(min_angle, c);
		min_angle = fminf(min_angle, d);

		return 2.f * min_angle / CUDART_PI;
	}

	

	// ######################################################################### //
	// ############################ Qualities ################################## //
	//                                                                           //
	//    These functions take the one-ring neighborhood of a vertex and         //
	//    calculate its entire quality.                                          //
	// ######################################################################### //
	

	// ######################################################################### //
	// ### Triangle Qualities: ################################################# //

	// ### AREA Quality for TriMeshes ########################################## //
	__host__ __device__ __forceinline__ float qualityAreaTri(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; ++k) {
			float ekp[2] = { oneRing[2 * k] - p[0], oneRing[2 * k + 1] - p[1] };
			float elp[2] = { oneRing[2 * (k + 1)] - p[0], oneRing[2 * (k + 1) + 1] - p[1] };

			float area = elp[1] * ekp[0] - elp[0] * ekp[1];

			q = fminf(q, area);
		}
		return q;
	}

	// ### SHAPE Quality for TriMeshes ######################################### //
	__host__ __device__ __forceinline__ float qualityShapeTri(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; ++k) {
			float v[3][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] } };
			q = fminf(q, metricMeanRatioTri(v));
		}

		return q;
	}

	// ### RIGHT_ANGLE Quality for TriMeshes ################################### //
	__host__ __device__ __forceinline__ float qualityRightAngleTri(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; ++k) {
			float v[3][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] } };
			q = fminf(q, metricRectangularityTri(v));
		}

		return q;
	}

	// ### MIN_ANGLE Quality for TriMeshes ##################################### //
	__host__ __device__ __forceinline__ float qualityMinAngleTri(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;
		for (int k = 0; k < n_oneRing - 1; ++k) {
			float v[3][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] } };
			q = fminf(q, metricMinAngleTri(v));
		}
		return q;
	}

	// ### RADIUS_RATIO Quality for TriMeshes ################################## //
	__host__ __device__ __forceinline__ float qualityRadiusRatioTri(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; ++k) {
			float v[3][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] } };
			q = fminf(q, metricRadiusRatioTri(v));
		}

		return q;
	}

	// ### MAX_ANGLE Quality for TriMeshes ##################################### //
	__host__ __device__ __forceinline__ float qualityMaxAngleTri(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; ++k) {
			float v[3][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] } };
			q = fminf(q, metricMaxAngleTri(v));
		}

		return q;
	}

	
	// ######################################################################### //
	// ### Quad Qualities: ##################################################### //

	// ### AREA Quality for QuadMeshes ######################################### //
	__host__ __device__ __forceinline__ float qualityAreaQuad(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; k += 2) {
			float e1[2] = { oneRing[2 * k] - p[0], oneRing[2 * k + 1] - p[1] };														// p --> k
			float e2[2] = { oneRing[2 * (k + 1)] - oneRing[2 * k], oneRing[2 * (k + 1) + 1] - oneRing[2 * k + 1] };					// k --> k + 1
			float e3[2] = { oneRing[2 * (k + 2)] - oneRing[2 * (k + 1)], oneRing[2 * (k + 2) + 1] - oneRing[2 * (k + 1) + 1] };		// k + 1 --> k + 2
			float e4[2] = { p[0] - oneRing[2 * (k + 2)], p[1] - oneRing[2 * (k + 2) + 1] };											// k + 2 --> p

			float area12 = e1[0] * e2[1] - e1[1] * e2[0];
			float area34 = e3[0] * e4[1] - e3[1] * e4[0];
			q = fminf(q, 2.f * (area12 + area34));
		}

		return q;
	}

	// ### SHAPE Quality for QuadMeshes ######################################## //
	__host__ __device__ __forceinline__ float qualityShapeQuad(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; k += 2) {
			float v[4][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] },{ oneRing[2 * (k + 2)], oneRing[2 * (k + 2) + 1] } };
			q = fminf(q, metricMeanRatioQuad(v));
		}

		return q;
	}

	// ### JACOBIAN Quality for QuadMeshes ##################################### //
	__host__ __device__ __forceinline__ float qualityJacobianQuad(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;

		for (int k = 0; k < n_oneRing - 1; k += 2) {
			float v[4][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] },{ oneRing[2 * (k + 2)], oneRing[2 * (k + 2) + 1] } };
			q = fminf(q, metricScaledJacobianQuad(v));
		}

		return q;
	}

	// ### MIN_ANGLE Quality for QuadMeshes #################################### //
	__host__ __device__ __forceinline__ float qualityMinAngleQuad(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2]) {
		float q = FLT_MAX;
		for (int k = 0; k < n_oneRing - 1; k += 2) {
			float v[4][2] = { { p[0], p[1] },{ oneRing[2 * k], oneRing[2 * k + 1] },{ oneRing[2 * (k + 1)], oneRing[2 * (k + 1) + 1] },{ oneRing[2 * (k + 2)], oneRing[2 * (k + 2) + 1] } };
			q = fminf(q, metricMinAngleQuad(v));
		}
		return q;
	}

	// ######################################################################### //
	// ### Quality Multiplexer ################################################# //
	
	__host__ __device__ __forceinline__ float quality(const int n_oneRing, const float oneRing[DMO_MAX_ONE_RING_SIZE], const float p[2], QualityCriterium q_crit) {
		switch (q_crit) {
		case MEAN_RATIO:
			#ifdef DMO_TRIANGLES
			return qualityShapeTri(n_oneRing, oneRing, p);
			#else
			return qualityShapeQuad(n_oneRing, oneRing, p);
			#endif
			break;
		case AREA:
			#ifdef DMO_TRIANGLES
			return qualityAreaTri(n_oneRing, oneRing, p);
			#else
			return qualityAreaQuad(n_oneRing, oneRing, p);
			#endif
			break;
		case RIGHT_ANGLE:
			#ifdef DMO_TRIANGLES
			return qualityRightAngleTri(n_oneRing, oneRing, p);
			#else
			// not implemented for quads
			#endif
			break;
		case JACOBIAN:
			#ifdef DMO_TRIANGLES
			// not implemented for triangles
			#else
			return qualityJacobianQuad(n_oneRing, oneRing, p);
			#endif	
			break;
		case MIN_ANGLE:
			#ifdef DMO_TRIANGLES
			return qualityMinAngleTri(n_oneRing, oneRing, p);
			#else 
			return qualityMinAngleQuad(n_oneRing, oneRing, p);
			#endif
			break;
		case RADIUS_RATIO:
			#ifdef DMO_TRIANGLES
			return qualityRadiusRatioTri(n_oneRing, oneRing, p);
			#else
			// not implemented for quads
			#endif
			break;
		case MAX_ANGLE:
			#ifdef DMO_TRIANGLES
			return qualityMaxAngleTri(n_oneRing, oneRing, p);
			#else
			// not implemented for quads
			#endif
			break;
		default:
			break;
		}
		return -1;
	}


	// ######################################################################### //
	// ### Quality Print Functions ############################################# //
		
	__host__ __device__ __forceinline__ float faceQuality(const float *vertexPos, const int *faceVec, const int n_face, QualityCriterium q_crit) {
		float q;

		switch (q_crit) {
		case MEAN_RATIO:
		{
			#ifdef DMO_TRIANGLES
			int v_id[3] = { faceVec[3 * n_face],  faceVec[3 * n_face + 1], faceVec[3 * n_face + 2] };
			float p[3][2];
			for (int i = 0; i < 3; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}
			q = metricMeanRatioTri(p);
			#else 
			int v_id[4] = { faceVec[4 * n_face],  faceVec[4 * n_face + 1], faceVec[4 * n_face + 2], faceVec[4 * n_face + 3] };
			float p[4][2];
			for (int i = 0; i < 4; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}

			q = metricMeanRatioQuad(p);
			#endif
			break;
		}
		case RIGHT_ANGLE:
		{
			#ifdef DMO_TRIANGLES
			int v_id[3] = { faceVec[3 * n_face],  faceVec[3 * n_face + 1], faceVec[3 * n_face + 2] };
			float p[3][2];
			for (int i = 0; i < 3; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}

			q = metricRectangularityTri(p);
			#else 
			printf("Right-Angle quality metric is not implemented for QuadMeshes.\n");
			#endif
			break;
		}
		case JACOBIAN:
		{
			#ifdef DMO_TRIANGLES
			printf("Jacobian quality metric is not implemented for TriMeshes.\n");
			#else
			int v_id[4] = { faceVec[4 * n_face],  faceVec[4 * n_face + 1], faceVec[4 * n_face + 2], faceVec[4 * n_face + 3] };
			float p[4][2];
			for (int i = 0; i < 4; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}
			q = metricScaledJacobianQuad(p);
			#endif
			break;
		}
		case MIN_ANGLE:
		{
			#ifdef DMO_TRIANGLES
			int v_id[3] = { faceVec[3 * n_face],  faceVec[3 * n_face + 1], faceVec[3 * n_face + 2] };
			float p[3][2];
			for (int i = 0; i < 3; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}

			q = metricMinAngleTri(p);
			#else
			int v_id[4] = { faceVec[4 * n_face],  faceVec[4 * n_face + 1], faceVec[4 * n_face + 2], faceVec[4 * n_face + 3] };
			float p[3][2];
			for (int i = 0; i < 4; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}

			q = metricMinAngleQuad(p);
			#endif
			break;
		}
		case RADIUS_RATIO:
		{
			#ifdef DMO_TRIANGLES
			int v_id[3] = { faceVec[3 * n_face],  faceVec[3 * n_face + 1], faceVec[3 * n_face + 2] };
			float p[3][2];
			for (int i = 0; i < 3; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}

			q = metricRadiusRatioTri(p);
			#else
			printf("Radius Ratio quality metric is not implemented for QuadMeshes.\n");
			#endif
			break;
		}
		case MAX_ANGLE:
		{
			#ifdef DMO_TRIANGLES
			int v_id[3] = { faceVec[3 * n_face],  faceVec[3 * n_face + 1], faceVec[3 * n_face + 2] };
			float p[3][2];
			for (int i = 0; i < 3; ++i) {
				p[i][0] = vertexPos[2 * v_id[i]];
				p[i][1] = vertexPos[2 * v_id[i] + 1];
			}

			q = metricMaxAngleTri(p);
			#else 
			printf("Max Angle quality metric is not implemented for QuadMeshes.\n");
			#endif
			break;
		}
		default:
			printf("Quality Metric unknown\n");
			break;
		}
		return q;
	}

	__global__ void printFaceQuality(const float *vertexPos, const int *faceVec, const int n_faces, QualityCriterium q_crit) {
		static int counter = 0;

		int q_vec[DMO_N_QUALITY_COLS] = { 0 };
		float q_min = FLT_MAX;

		for (int i = 0; i < n_faces; ++i) {
			float q = faceQuality(vertexPos, faceVec, i, q_crit);

			if (q > 1.f) q = 1.f;
			else if (q < 0.f) q = 0.f;

			int index = (int)(q*DMO_N_QUALITY_COLS);
			q_vec[index] += 1;
			q_min = fminf(q_min, q);
		}

		printf("%3d: ", counter++);

		for (int i = 0; i < DMO_N_QUALITY_COLS; ++i) {
			if (q_vec[i] != 0)
				printf("%4d | ", q_vec[i]);
			else
				printf("     | ");
		}
		printf("| q_min = %1.6f", q_min);
		printf("\n");
	}
	
	__global__ void printFaceQuality(const float *vertexPos, const int *faceVec, const int n_faces, QualityCriterium q_crit, float *q_min_vec, float *q_avg_vec) {
		static int counter = 0;

		float q_min = FLT_MAX;
		float q_avg = 0;

		for (int i = 0; i < n_faces; ++i) {
			float q = faceQuality(vertexPos, faceVec, i, q_crit);
			q_min = fminf(q_min, q);
			q_avg += q;
		}
		q_avg /= n_faces;

		q_min_vec[counter] = q_min;
		q_avg_vec[counter++] = q_avg;
	}


	// ######################################################################### //
	// ### DMO Algorithm ####################################################### //
	// ######################################################################### //

	__global__ void optimizeHierarchical(int* coloredVertexIDs, const int cOff, const Vertex* vertices, float* vertexPos, int* oneRingVec, const float affineFactor, QualityCriterium q_crit, const float grid_scale) {
		const int i1 = threadIdx.x / DMO_NQ;
		const int j1 = threadIdx.x % DMO_NQ;

		const int i2 = (threadIdx.x + DMO_NQ * DMO_NQ / 2) / DMO_NQ;
		const int j2 = (threadIdx.x + DMO_NQ * DMO_NQ / 2) % DMO_NQ;

		const int vid = coloredVertexIDs[cOff + blockIdx.x];
		const Vertex& v = vertices[vid];

		float q = -FLT_MAX;

		__shared__ float xPos, yPos;
		__shared__ float maxDistx, maxDisty;

		__shared__ my_atomics argMaxVal;
		argMaxVal.floats[0] = -FLT_MAX;
		argMaxVal.ints[1] = DMO_NQ*DMO_NQ;

		__shared__ float oneRing[DMO_MAX_ONE_RING_SIZE];

		// min/max search + loading oneRing
		if (threadIdx.x == 0) {
			for (int k = 0; k < v.n_oneRing - 1; ++k) {
				float oneRingX = vertexPos[2 * oneRingVec[v.oneRingID + k]];
				float oneRingY = vertexPos[2 * oneRingVec[v.oneRingID + k] + 1];
				oneRing[2 * k] = oneRingX;
				oneRing[2 * k + 1] = oneRingY;

				float xDist = abs(vertexPos[2 * v.id] - oneRingX);
				float yDist = abs(vertexPos[2 * v.id + 1] - oneRingY);

				maxDistx = fmaxf(maxDistx, xDist);
				maxDisty = fmaxf(maxDisty, yDist);
			}

			// set xmaxmin...
			maxDistx = grid_scale * maxDistx;
			maxDisty = grid_scale * maxDisty;
			
			oneRing[2 * v.n_oneRing - 2] = vertexPos[2 * oneRingVec[v.oneRingID + v.n_oneRing - 1]];
			oneRing[2 * v.n_oneRing - 1] = vertexPos[2 * oneRingVec[v.oneRingID + v.n_oneRing - 1] + 1];

			xPos = vertexPos[2 * v.id];
			yPos = vertexPos[2 * v.id + 1];
		}

		// start depth iteration
		float depth_scale = grid_scale;
		float argMax = 0;
		for (int depth = 0; depth < DMO_DEPTH; ++depth) {

			float xMax, xMin, yMax, yMin;
			xMax = xPos + depth_scale * maxDistx;
			xMin = xPos - depth_scale * maxDistx;
			yMax = yPos + depth_scale * maxDisty;
			yMin = yPos - depth_scale * maxDisty;

			float pCurrent[2] = { xPos, yPos };

			float pos_i1 = affineFactor * (i1 * xMin + (DMO_NQ - 1 - i1) * xMax);
			float pos_j1 = affineFactor * (j1 * yMin + (DMO_NQ - 1 - j1) * yMax);
			float pos_i2 = affineFactor * (i2 * xMin + (DMO_NQ - 1 - i2) * xMax);
			float pos_j2 = affineFactor * (j2 * yMin + (DMO_NQ - 1 - j2) * yMax);

			float p1[2] = { pos_i1, pos_j1 };
			float q1 = quality(v.n_oneRing, oneRing, p1, q_crit);
			float p2[2] = { pos_i2, pos_j2 };
			float q2 = quality(v.n_oneRing, oneRing, p2, q_crit);

			if (q1 > q2) {
				q = q1;
				argMax = 1;
			}
			else {
				q = q2;
				argMax = 2;
			}

			my_atomicArgMax(&(argMaxVal.ulong), q, i1 * DMO_NQ + j1);

			float qOld = quality(v.n_oneRing, oneRing, pCurrent, q_crit);
			if (i1 * DMO_NQ + j1 == argMaxVal.ints[1] && qOld < q) {
				if (argMax == 1) {
					xPos = pos_i1;
					yPos = pos_j1;
				}
				else {
					xPos = pos_i2;
					yPos = pos_j2;
				}
			}

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}


		// set new position if it is better than the old one
		float pOld[2] = { vertexPos[2 * v.id] , vertexPos[2 * v.id + 1] };
		if (i1 * DMO_NQ + j1 == argMaxVal.ints[1] && quality(v.n_oneRing, oneRing, pOld, q_crit) < q) {
			vertexPos[2 * v.id] = xPos;
			vertexPos[2 * v.id + 1] = yPos;
		}
	}


	// ######################################################################### //
	// ############################# OpenMesh ################################## //
	// ######################################################################### //

	
	// ######################################################################### //
	// ### Both Meshes Version ################################################# //
	
	void copyOpenMeshData(DMOMesh& mesh, float* vertexPos, Vertex* vertices, int* oneRingVec) {
		int interior_counter = 0;
		int oneRing_counter = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			DMOMesh::Point p = mesh.point(*v_it);

			vertexPos[2 * v_it->idx()] = p[0];
			vertexPos[2 * v_it->idx() + 1] = p[1];


			if (!mesh.is_boundary(*v_it)) {
				// fill vertex struct

				Vertex& v = vertices[interior_counter];
				v.id = v_it->idx();

				v.n_oneRing = 0;
				for (auto voh_it = mesh.voh_iter(*v_it); voh_it.is_valid(); ++voh_it) {
					++v.n_oneRing;
				#ifndef DMO_TRIANGLES
					if (!mesh.is_boundary(*voh_it)) ++v.n_oneRing;
				#endif
				}
				++v.n_oneRing;

				v.oneRingID = oneRing_counter;

				DMOMesh::HalfedgeHandle heh = *(mesh.voh_iter(*v_it));
				DMOMesh::HalfedgeHandle heh_init = heh;

				do {
					oneRingVec[oneRing_counter++] = mesh.to_vertex_handle(heh).idx();
					heh = mesh.next_halfedge_handle(heh);
				#ifndef DMO_TRIANGLES
					oneRingVec[oneRing_counter++] = mesh.to_vertex_handle(heh).idx();
					heh = mesh.next_halfedge_handle(heh);
				#endif
					heh = mesh.next_halfedge_handle(heh);
					heh = mesh.opposite_halfedge_handle(heh);
				} while (heh.idx() != heh_init.idx());

				oneRingVec[oneRing_counter] = mesh.to_vertex_handle(heh).idx();		// close one-ring
				++oneRing_counter;

				++interior_counter;
			}
		}
	}

	inline void createColoring(DMOMesh& mesh, const int n_free_vertices, int** coloredVertexIDs, std::vector<int>& colorOffset) {
		// create coloring scheme
		std::vector<int>colorScheme(mesh.n_vertices(), -1);
		int colorSchemeIt = 0;

		// set boundarys to a value that can be ignored
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			if (mesh.is_boundary(*v_it)) {
				colorScheme[v_it->idx()] = -2;
			}
		}

		while (std::find(colorScheme.begin(), colorScheme.end(), -1) != colorScheme.end()) {
			for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
				// check if vertex is already colored
				if (colorScheme[v_it->idx()] != -1) { continue; }

				bool neighborIsCurrent = false;
			#ifdef DMO_TRIANGLES
				for (auto vv_it = mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
					if (colorScheme[vv_it->idx()] == colorSchemeIt) {
						neighborIsCurrent = true;
						break;
					}
				}
			#else 
				for (auto voh_it = mesh.voh_iter(*v_it); voh_it.is_valid(); ++voh_it) {
					DMOMesh::VertexHandle vh1 = mesh.to_vertex_handle(*voh_it);
					DMOMesh::VertexHandle vh2 = mesh.to_vertex_handle(mesh.next_halfedge_handle(*voh_it));
					if (colorScheme[vh1.idx()] == colorSchemeIt || colorScheme[vh2.idx()] == colorSchemeIt) {
						neighborIsCurrent = true;
						break;
					}
				}
			#endif
				// check if a neighboring vertex is already in this color
				if (neighborIsCurrent) { continue; }
				colorScheme[v_it->idx()] = colorSchemeIt;
			}
			++colorSchemeIt;
		}

		int n_colors = *(std::max_element(colorScheme.begin(), colorScheme.end())) + 1;

		std::vector<int> n_color_vecs(n_colors, 0);
		for (int i = 0; i < colorScheme.size(); ++i) {
			if (colorScheme[i] > -1)
				++n_color_vecs[colorScheme[i]];
		}

		*coloredVertexIDs = new int[n_free_vertices];

		colorOffset = std::vector<int>(n_colors + 1, 0);
		for (int i = 1; i < n_colors; ++i) {
			colorOffset[i] = colorOffset[i - 1] + n_color_vecs[i - 1];
		}
		// mark the end of the colored-vertices vector:
		colorOffset[n_colors] = n_free_vertices;

		// add vertex ids
		std::vector<int>colorCounter(n_colors, 0);
		int interior_counter = 0;
		for (int i = 0; i < colorScheme.size(); ++i) {
			if (colorScheme[i] < 0) { continue; }
			(*coloredVertexIDs)[colorOffset[colorScheme[i]] + colorCounter[colorScheme[i]]++] = interior_counter++;
		}
	}


	// ######################################################################### //
	// ### TriMesh version ##################################################### //

#ifdef DMO_TRIANGLES
	void discreteMeshOptimization(DMOMesh& mesh, QualityCriterium q_crit = MEAN_RATIO, const float grid_scale = 0.5f, int n_iter = 100) {

		Stopwatch sw;
		int n_free_vertices = 0;
		int oneRingVecLength = 0;
#pragma omp parallel for reduction(+:n_free_vertices,oneRingVecLength)
		for (int i = 0; i < mesh.n_vertices(); ++i) {
			DMOMesh::VertexHandle vh = mesh.vertex_handle(i);
			if (mesh.is_boundary(vh)) { continue; }
			++n_free_vertices;

			for (auto vv_it = mesh.vv_iter(vh); vv_it.is_valid(); ++vv_it) { ++oneRingVecLength; }
			// additional count so that last element is again the first element
			++oneRingVecLength;
		}

		// convert OpenMesh to a basic structure
		float* vertexPos = new float[2 * mesh.n_vertices()];
		Vertex* vertices = new Vertex[n_free_vertices];
		int* oneRingVec = new int[oneRingVecLength];

		float* d_vertexPos;
		Vertex* d_vertices;
		int* d_oneRingVec;
		int* d_coloredVertexIDs;

		int* coloredVertexIDs;
		std::vector<int> colorOffset;


		#pragma omp parallel sections num_threads(2)
		{
			#pragma omp section
			{
				gpuErrchk(cudaMalloc((void**)&d_vertexPos, 2 * mesh.n_vertices() * sizeof(float)));
				gpuErrchk(cudaMalloc((void**)&d_vertices, n_free_vertices * sizeof(Vertex)));
				gpuErrchk(cudaMalloc((void**)&d_oneRingVec, oneRingVecLength * sizeof(int)));
				gpuErrchk(cudaMalloc((void**)&d_coloredVertexIDs, n_free_vertices * sizeof(int)));

				createColoring(mesh, n_free_vertices, &coloredVertexIDs, colorOffset);
				gpuErrchk(cudaMemcpyAsync(d_coloredVertexIDs, coloredVertexIDs, n_free_vertices * sizeof(int), cudaMemcpyHostToDevice));
			}
			#pragma omp section 
			{
				copyOpenMeshData(mesh, vertexPos, vertices, oneRingVec);
			}
		}

		gpuErrchk(cudaMemcpyAsync(d_vertexPos, vertexPos, 2 * mesh.n_vertices() * sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpyAsync(d_vertices, vertices, n_free_vertices * sizeof(Vertex), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpyAsync(d_oneRingVec, oneRingVec, oneRingVecLength * sizeof(int), cudaMemcpyHostToDevice));

		int n_colors = colorOffset.size() - 1;


		// face vector (only needed for quality evaluation)
#if DMO_PRINT_QUALITY
		int* faceVec = new int[mesh.n_faces() * 3];
		for (int i = 0; i < mesh.n_faces(); ++i) {
			DMOMesh::FaceHandle fh = mesh.face_handle(i);
			int vertex_counter = 0;
			for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
				faceVec[3 * i + vertex_counter++] = fv_it->idx();
			}
		}
		int* d_faceVec;
		gpuErrchk(cudaMalloc((void**)&d_faceVec, 3 * mesh.n_faces() * sizeof(int)));
		gpuErrchk(cudaMemcpy(d_faceVec, faceVec, 3 * mesh.n_faces() * sizeof(int), cudaMemcpyHostToDevice));
#endif // DMO_PRINT_QUALITY

		const float affineFactor = 1.f / (float)(DMO_NQ - 1);


#if DMO_PRINT_QUALITY
		// q_min_vec for printing
		float *q_min_vec, *q_avg_vec;
		cudaMallocManaged(&q_min_vec, (n_iter + 1) * sizeof(float));
		cudaMallocManaged(&q_avg_vec, (n_iter + 1) * sizeof(float));

		printf("    ");
		for (int i = 0; i < DMO_N_QUALITY_COLS; ++i) {
			printf("<%1.3f|", (float)(i + 1) / (float)DMO_N_QUALITY_COLS);
		}
		printf("\n\n");
		printFaceQuality <<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit);
		printFaceQuality <<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit, q_min_vec, q_avg_vec);
#endif // DMO_PRINT_QUALITY

		for (int i = 0; i < n_iter; ++i) {
			for (int cid = 0; cid < n_colors; ++cid) {
				optimizeHierarchical<<<colorOffset[cid + 1] - colorOffset[cid], DMO_NQ * DMO_NQ / 2 >>>(d_coloredVertexIDs, colorOffset[cid], d_vertices, d_vertexPos, d_oneRingVec, affineFactor, q_crit, grid_scale);
			}
#if DMO_PRINT_QUALITY
			cudaDeviceSynchronize();
			printFaceQuality<<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit);
			printFaceQuality<<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit, q_min_vec, q_avg_vec);
#endif // DMO_PRINT_QUALITY
		}

#if DMO_PRINT_QUALITY
		cudaDeviceSynchronize();
		std::string ofs_name = "../output.txt";
		std::ofstream ofs;
		ofs.open(ofs_name);
		for (int i = 0; i < n_iter + 1; ++i) {
			ofs << i << " " << q_min_vec[i] << " " << q_avg_vec[i] << std::endl;
		}
		ofs.close();
#endif // DMO_PRINT_QUALITY

		cudaMemcpy(vertexPos, d_vertexPos, 2 * mesh.n_vertices() * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_vertexPos);
		cudaFree(d_vertices);
		cudaFree(d_oneRingVec);
		cudaFree(d_coloredVertexIDs);

		delete[] vertices;
		delete[] oneRingVec;
		delete[] coloredVertexIDs;

#if DMO_PRINT_QUALITY
		delete[] faceVec;

		cudaFree(d_faceVec);
		cudaFree(q_min_vec);
		cudaFree(q_avg_vec);
#endif // DMO_PRINT_QUALITY

		// write vertex positions back to mesh
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			int id = v_it->idx();
			DMOMesh::Point p = { vertexPos[2 * id], vertexPos[2 * id + 1], 0.f };
			mesh.set_point(*v_it, p);
		}

		delete[] vertexPos;
	}
#endif

	// ######################################################################### //
	// ### QuadMesh version #################################################### //
	
#ifndef DMO_TRIANGLES
	void discreteMeshOptimization(DMOMesh& mesh, QualityCriterium q_crit = MEAN_RATIO, const float grid_scale = 0.5f, int n_iter = 100) {

		int n_free_vertices = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) { if (!mesh.is_boundary(*v_it)) ++n_free_vertices; }

		int oneRingVecLength = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			if (mesh.is_boundary(*v_it)) { continue; }
			for (auto voh_it = mesh.voh_iter(*v_it); voh_it.is_valid(); ++voh_it) {
				++oneRingVecLength;
				if (!mesh.is_boundary(*voh_it)) ++oneRingVecLength;
			}
			// additional count s.th. last element is again the first element
			++oneRingVecLength;
		}

		// convert OpenMesh to a basic structure
		float* vertexPos = new float[2 * mesh.n_vertices()];
		Vertex* vertices = new Vertex[n_free_vertices];
		int* oneRingVec = new int[oneRingVecLength];
		int* faceVec = new int[mesh.n_faces() * 4];

		Stopwatch sw;
		copyOpenMeshData(mesh, vertexPos, vertices, oneRingVec);


		// coloring
		int* coloredVertexIDs;
		std::vector<int> colorOffset;
		createColoring(mesh, n_free_vertices, &coloredVertexIDs, colorOffset);
		int n_colors = colorOffset.size() - 1;


		// face vector (only needed for quality evaluation)
		for (int i = 0; i < mesh.n_faces(); ++i) {
			TriMesh::FaceHandle fh = mesh.face_handle(i);
			int vertex_counter = 0;
			for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
				faceVec[4 * i + vertex_counter++] = fv_it->idx();
			}
		}


		const float affineFactor = 1.f / (float)(DMO_NQ - 1);

		float* d_vertexPos;
		Vertex* d_vertices;
		int* d_oneRingVec;
		int* d_coloredVertexIDs;
		int* d_faceVec;


		gpuErrchk(cudaMalloc((void**)&d_vertexPos, 2 * mesh.n_vertices() * sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_vertices, n_free_vertices * sizeof(Vertex)));
		gpuErrchk(cudaMalloc((void**)&d_oneRingVec, oneRingVecLength * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_faceVec, 4 * mesh.n_faces() * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_coloredVertexIDs, n_free_vertices * sizeof(int)));

		gpuErrchk(cudaMemcpy(d_vertexPos, vertexPos, 2 * mesh.n_vertices() * sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_vertices, vertices, n_free_vertices * sizeof(Vertex), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_oneRingVec, oneRingVec, oneRingVecLength * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_faceVec, faceVec, 4 * mesh.n_faces() * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_coloredVertexIDs, coloredVertexIDs, n_free_vertices * sizeof(int), cudaMemcpyHostToDevice));

#if DMO_PRINT_QUALITY
		// q_min_vec for printing
		float *q_min_vec, *q_avg_vec;
		cudaMallocManaged(&q_min_vec, (n_iter + 1) * sizeof(float));
		cudaMallocManaged(&q_avg_vec, (n_iter + 1) * sizeof(float));

		printf("    ");
		for (int i = 0; i < DMO_N_QUALITY_COLS; ++i) {
			printf("<%1.3f|", (float)(i + 1) / (float)DMO_N_QUALITY_COLS);
		}
		printf("\n\n");
		printFaceQuality<<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit);
		printFaceQuality<<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit, q_min_vec, q_avg_vec);
#endif // DMO_PRINT_QUALITY

		cudaDeviceSynchronize();
		sw.start();
		for (int i = 0; i < n_iter; ++i) {
			for (int cid = 0; cid < n_colors; ++cid) {
				optimizeHierarchical<<<colorOffset[cid + 1] - colorOffset[cid], DMO_NQ * DMO_NQ / 2 >>>(d_coloredVertexIDs, colorOffset[cid], d_vertices, d_vertexPos, d_oneRingVec, affineFactor, q_crit, grid_scale);
			}
#if DMO_PRINT_QUALITY
			cudaDeviceSynchronize();
			printFaceQuality<<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit);
			printFaceQuality<<<1, 1 >>>(d_vertexPos, d_faceVec, mesh.n_faces(), q_crit, q_min_vec, q_avg_vec);
#endif // DMO_PRINT_QUALITY
		}
		cudaDeviceSynchronize();
		sw.stop();
		std::cout << "DMO runtime: " << sw.runtimeStr() << std::endl;

#if DMO_PRINT_QUALITY
		cudaDeviceSynchronize();
		std::string ofs_name = "../output.txt";
		std::ofstream ofs(ofs_name);
		for (int i = 0; i < n_iter + 1; ++i) {
			ofs << i << " " << q_min_vec[i] << " " << q_avg_vec[i] << std::endl;
		}
		ofs.close();
#endif // DMO_PRINT_QUALITY

		cudaMemcpy(vertexPos, d_vertexPos, 2 * mesh.n_vertices() * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_vertexPos);
		cudaFree(d_vertices);
		cudaFree(d_oneRingVec);
		cudaFree(d_faceVec);
		cudaFree(d_coloredVertexIDs);

#if DMO_PRINT_QUALITY
		cudaFree(q_min_vec);
		cudaFree(q_avg_vec);
#endif // DMO_PRINT_QUALITY

		// write vertex positions back to mesh
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			int id = v_it->idx();
			TriMesh::Point p = { vertexPos[2 * id], vertexPos[2 * id + 1], 0.f };
			mesh.set_point(*v_it, p);
		}

		delete[] vertexPos;
		delete[] vertices;
		delete[] oneRingVec;
		delete[] faceVec;
		delete[] coloredVertexIDs;
	}
#endif

	// ######################################################################### //
	// ### CPU version ######################################################### //

	inline void optimizeHierarchical(const int vid, const Vertex* vertices, float* vertexPos, int* oneRingVec, const float affineFactor, QualityCriterium q_crit, const float grid_scale) {

		const Vertex& v = vertices[vid];

		float xPos, yPos;
		float maxDistx = 0, maxDisty = 0;

		float oneRing[DMO_MAX_ONE_RING_SIZE];

		// min/max search + loading oneRing
		for (int k = 0; k < v.n_oneRing - 1; ++k) {
			float oneRingX = vertexPos[2 * oneRingVec[v.oneRingID + k]];
			float oneRingY = vertexPos[2 * oneRingVec[v.oneRingID + k] + 1];
			oneRing[2 * k] = oneRingX;
			oneRing[2 * k + 1] = oneRingY;

			float xDist = abs(vertexPos[2 * v.id] - oneRingX);
			float yDist = abs(vertexPos[2 * v.id + 1] - oneRingY);

			maxDistx = fmaxf(maxDistx, xDist);
			maxDisty = fmaxf(maxDisty, yDist);
		}

		// set xmaxmin...
		maxDistx = grid_scale * maxDistx;
		maxDisty = grid_scale * maxDisty;

		oneRing[2 * v.n_oneRing - 2] = vertexPos[2 * oneRingVec[v.oneRingID + v.n_oneRing - 1]];
		oneRing[2 * v.n_oneRing - 1] = vertexPos[2 * oneRingVec[v.oneRingID + v.n_oneRing - 1] + 1];

		xPos = vertexPos[2 * v.id];
		yPos = vertexPos[2 * v.id + 1];

		float pOld[2] = { xPos, yPos };
		float q = quality(v.n_oneRing, oneRing, pOld, q_crit);

		// start depth iteration
		float depth_scale = grid_scale;
		for (int depth = 0; depth < DMO_DEPTH; ++depth) {

			float xMax, xMin, yMax, yMin;
			xMax = xPos + depth_scale * maxDistx;
			xMin = xPos - depth_scale * maxDistx;
			yMax = yPos + depth_scale * maxDisty;
			yMin = yPos - depth_scale * maxDisty;

#pragma omp parallel for
			for (int i = 0; i < DMO_NQ; ++i) {
				float pos_i = affineFactor * (i * xMin + (DMO_NQ - 1 - i) * xMax);

				for (int j = 0; j < DMO_NQ; ++j) {
					float pos_j = affineFactor * (j * yMin + (DMO_NQ - 1 - j) * yMax);

					float pCurrent[2] = { pos_i, pos_j };
					float qCurrent = quality(v.n_oneRing, oneRing, pCurrent, q_crit);

					if (qCurrent > q) {
						xPos = pos_i;
						yPos = pos_j;
						q = qCurrent;
					}
				}
			}

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}


		// set new position if it is better than the old one
		vertexPos[2 * v.id] = xPos;
		vertexPos[2 * v.id + 1] = yPos;
	}

#ifdef DMO_TRIANGLES
	void discreteMeshOptimizationCPU(DMOMesh& mesh, QualityCriterium q_crit = MEAN_RATIO, const float grid_scale = 0.5f, int n_iter = 100) {

		int n_free_vertices = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) { if (!mesh.is_boundary(*v_it)) ++n_free_vertices; }
		//printf("N free vertices = %d\n", n_free_vertices);

		int oneRingVecLength = 0;
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			if (mesh.is_boundary(*v_it)) { continue; }
			for (auto vv_it = mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) { ++oneRingVecLength; }
			++oneRingVecLength;		// additional count s.th. last element is again the first element
		}

		// convert OpenMesh to a basic structure
		float* vertexPos = new float[2 * mesh.n_vertices()];
		Vertex* vertices = new Vertex[n_free_vertices];
		int* oneRingVec = new int[oneRingVecLength];

		Stopwatch sw;
		copyOpenMeshData(mesh, vertexPos, vertices, oneRingVec);

		const float affineFactor = 1.f / (float)(DMO_NQ - 1);

		sw.start();
		for (int i = 0; i < n_iter; ++i) {
			for (int vid = 0; vid < n_free_vertices; ++vid) {
				optimizeHierarchical(vid, vertices, vertexPos, oneRingVec, affineFactor, q_crit, grid_scale);
			}
		}
		sw.stop();
		std::cout << "DMO runtime: " << sw.runtimeStr() << std::endl;

		// write vertex positions back to mesh
		for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
			int id = v_it->idx();
			TriMesh::Point p = { vertexPos[2 * id], vertexPos[2 * id + 1], 0.f };
			mesh.set_point(*v_it, p);
		}

		delete[] vertexPos;
		delete[] vertices;
		delete[] oneRingVec;
	}
#endif

}