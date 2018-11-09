#include "DiscreteMeshOptimization.h"

namespace DMO {
	
	// ##################################################################### //
	// ### CPU functions ################################################### //
	// ##################################################################### //


	inline void optimizeHierarchicalCPU(PolyMesh& mesh, const PolyMesh::VertexHandle& vh) {
		const size_t N = 8;
		const float grid_scale = 0.5f;
		const int depth_max = 3;

		PolyMesh::Point p_opt = mesh.point(vh);

		// collect one-ring neighborhood
		std::vector<Eigen::Vector2f> oneRing;
		PolyMesh::HalfedgeHandle heh = mesh.halfedge_handle(vh);
		PolyMesh::HalfedgeHandle heh_init = heh;
		do {
			PolyMesh::Point p = mesh.point(mesh.to_vertex_handle(heh));
			oneRing.push_back({ p[0],p[1] });
			heh = mesh.next_halfedge_handle(heh);
			p = mesh.point(mesh.to_vertex_handle(heh));
			oneRing.push_back({ p[0],p[1] });
			heh = mesh.next_halfedge_handle(heh);
			heh = mesh.opposite_halfedge_handle(mesh.next_halfedge_handle(heh));
		} while (heh != heh_init);
		oneRing.push_back(oneRing[0]);		// close ring by making first element also the last element

											// pre-compute length of one-ring edges
		std::vector<float> oneRingLength(oneRing.size());
		for (size_t i = 0; i < oneRing.size() - 1; ++i) {
			oneRingLength[i] = (oneRing[i] - oneRing[i + 1]).squaredNorm();
		}

		// compute grid-size
		float maxDistx = 0.f;
		float maxDisty = 0.f;
		for (size_t i = 0; i < oneRing.size(); ++i) {
			float xDist = std::fabsf(p_opt[0] - oneRing[i][0]);
			float yDist = std::fabsf(p_opt[1] - oneRing[i][1]);

			maxDistx = fmaxf(maxDistx, xDist);
			maxDisty = fmaxf(maxDisty, yDist);
		}

		maxDistx = grid_scale * maxDistx;
		maxDisty = grid_scale * maxDisty;


		float affineFactor = 1.f / (float)(N - 1);
		float depth_scale = grid_scale;

		// compute quality
		for (int depth = 0; depth < depth_max; ++depth) {
			std::vector<float> q(N*N, 1);

			float xMax, xMin, yMax, yMin;
			xMax = p_opt[0] + depth_scale * maxDistx;
			xMin = p_opt[0] - depth_scale * maxDistx;
			yMax = p_opt[1] + depth_scale * maxDisty;
			yMin = p_opt[1] - depth_scale * maxDisty;

	#pragma omp parallel for
			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < N; ++j) {

					Eigen::Vector2f p = { affineFactor * (i * xMin + (N - 1 - i) * xMax), affineFactor * (j * yMin + (N - 1 - j) * yMax) };

					// find minimal quality at node (i,j)
					for (int k = 0; k < oneRing.size() - 1; k += 2) {
						Eigen::Vector2f points[4] = { p, oneRing[k], oneRing[k + 1], oneRing[k + 2] };

						float quality = metricMeanRatioQuad(points);

						q[i * N + j] = std::min(q[i * N + j], quality);
					}
				}
			}


			// find max of q
			size_t iOpt = 0;
			size_t jOpt = 0;
			float qOpt = 0;

			for (size_t i = 0; i < N; ++i) {
				for (size_t j = 0; j < N; ++j) {
					if (q[i * N + j] > qOpt) {
						iOpt = i;
						jOpt = j;
						qOpt = q[i * N + j];
					}
				}
			}

			p_opt = { affineFactor* (iOpt * xMin + (N - 1 - iOpt) * xMax), affineFactor * (jOpt * yMin + (N - 1 - jOpt) * yMax), 0.f };

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (N - 1));
		}

		mesh.set_point(vh, p_opt);
	}

	void discreteMeshOptimizationCPU(PolyMesh& mesh, const int qualityCriterium, const float gridScale, int n_iter) {
		// do n_iter smoothing iterations
		for (size_t i = 0; i < n_iter; ++i) {
			// smoothen every vertex of mesh except the boundary
			for (auto vh_it = mesh.vertices_begin(); vh_it != mesh.vertices_end(); ++vh_it) {
				if (mesh.is_boundary(*vh_it)) continue;
				optimizeHierarchicalCPU(mesh, *vh_it);
			}
		}
	}


	// ##################################################################### //
	// ### Metric functions ################################################ //
	// ##################################################################### //

	float metricMeanRatioTri(TriMesh & mesh, const TriMesh::FaceHandle & fh)
	{
		// From the paper "R. Rangarajan, A.J. Lew: Directional Vertex Relaxation"
		std::vector<Eigen::Vector2f> points;
		for (auto fv_iter = mesh.fv_iter(fh); fv_iter.is_valid(); ++fv_iter) {
			TriMesh::Point p = mesh.point(*fv_iter);
			Eigen::Vector2f currentPoint;
			currentPoint[0] = p[0];
			currentPoint[1] = p[1];
			points.push_back(currentPoint);
		}

		Eigen::Vector2f e12 = points[0] - points[1];
		Eigen::Vector2f e23 = points[1] - points[2];
		Eigen::Vector2f e13 = points[0] - points[2];

		double l = e12.squaredNorm() + e23.squaredNorm() + e13.squaredNorm();
		double area = 0.5 * (e23[1] * e13[0] - e23[0] * e13[1]);

		return 4 * std::sqrt(3) * area / l;
	}

	float metricMeanRatioQuad(PolyMesh & mesh, const PolyMesh::FaceHandle & fh)
	{
		int i = 0;
		Eigen::Vector2f points[4];
		for (auto fv_iter = mesh.fv_iter(fh); fv_iter.is_valid(); ++fv_iter) {
			PolyMesh::Point p = mesh.point(*fv_iter);
			points[i][0] = p[0];
			points[i][1] = p[1];
			++i;
		}

		return metricMeanRatioQuad(points);
	}
	
	float metricMeanRatioQuad(Eigen::Vector2f points[4])
	{
		float e[4][2];
		float e_length_squared[4];
		for (size_t i = 0; i < 4; ++i) {

			int j = (i + 1) % 4;
			e[i][0] = points[j][0] - points[i][0];
			e[i][1] = points[j][1] - points[i][1];

			e_length_squared[i] = e[i][0] * e[i][0] + e[i][1] * e[i][1];
		}

		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2] + e_length_squared[3];
		float area1 = e[0][0] * e[1][1] - e[0][1] * e[1][0];
		float area2 = e[2][0] * e[3][1] - e[2][1] * e[3][0];

		return 2.f * (area1 + area2) / l;
	}


	// ##################################################################### //
	// ### Print Quality functions ######################################### //
	// ##################################################################### //

	void printQuality(TriMesh& mesh, std::ofstream& ofs) {
		std::vector<float> q(mesh.n_faces());

		for (int i = 0; i < mesh.n_faces(); ++i) {
			q[i] = metricMeanRatioTri(mesh, mesh.face_handle(i));
		}

		std::sort(q.begin(), q.end());

		for (int i = 0; i < mesh.n_faces(); ++i) {
			ofs << q[i] << " ";
		}
		ofs << std::endl;
	}

	void displayQuality(TriMesh& mesh, int n_cols) {
		std::vector<float> q_vec(n_cols, 0);

		float q_min = FLT_MAX;
		// measure quality
		for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
			float q = metricMeanRatioTri(mesh, *f_it);
			q_min = fminf(q_min, q);
			q_vec[size_t(q * 10 - 0.0001)] += 1;
		}

		for (size_t i = 0; i < q_vec.size(); ++i) {
			std::cout << "q:    " << (float)i / (float)q_vec.size() << " - " << (float)(i + 1) / (float)q_vec.size() << " = " << q_vec[i] << std::endl;
		}
		std::cout << "q_min = " << q_min << std::endl;
	}

	void displayQuality(PolyMesh& mesh, int n_cols) {
		std::vector<float> q_vec(n_cols, 0);

		// measure quality
		for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
			float q = metricMeanRatioQuad(mesh, *f_it);
			q_vec[size_t(q * 10 - 0.0001)] += 1;
		}

		for (size_t i = 0; i < q_vec.size(); ++i) {
			std::cout << "q:    " << (float)i / (float)q_vec.size() << " - " << (float)(i + 1) / (float)q_vec.size() << " = " << q_vec[i] << std::endl;
		}
	}
}