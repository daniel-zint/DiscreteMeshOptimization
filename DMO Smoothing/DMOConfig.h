#pragma once

#include "OpenMeshConfig.h"

/* Keep DMO_NQ = 8 for two dimensional meshes! This value was chosen because it gives optimal
performance considering a warp-size of 32 because NQ = 8 results in 8 * 8 = 64 nodes
which is double the warp size. Each vertex is computed using one warp where each warp
computes two grid nodes.
Another implementation used 2 warps for one grid but it was slower as syncthreads is
too expensive.
*/

// Size of Quality Mesh
#define DMO_NQ 8
// number of refinement steps within DMO
#define DMO_DEPTH 3
// double the maximal number of allowed vertices on the one-ring neighborhood
#define DMO_MAX_ONE_RING_SIZE 32
// Maximal number of allowed vertices on the one-ring neighborhood in 3D
#define DMO_MAX_ONE_RING_SIZE_3D 360

// For quality output
#define DMO_N_QUALITY_COLS 10
// Set this value to print quality
#define DMO_PRINT_QUALITY 1

// comment to use QuadMeshes (PolyMeshes) and uncomment to use TriMeshes
#define DMO_TRIANGLES

#ifdef DMO_TRIANGLES
typedef TriMesh DMOMesh;
#else
typedef PolyMesh DMOMesh;
#endif