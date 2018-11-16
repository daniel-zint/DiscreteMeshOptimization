/**
 * Copyright (C) 2018 by Daniel Zint and Philipp Guertler
 * This file is part of Discrete Mesh Optimization DMO
 * Some rights reserved. See LICENCE.
 */

#pragma once

#include <OpenMesh\Core\IO\MeshIO.hh>
#include <OpenMesh\Core\Mesh\TriMesh_ArrayKernelT.hh>
#include <OpenMesh\Core\Mesh\PolyMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<> TriMesh;
typedef OpenMesh::PolyMesh_ArrayKernelT<> PolyMesh;