#pragma once
#define _XKEYCHECK_H

#if defined (_MSC_VER)  // Visual studio
#define thread_local __declspec( thread )
#elif defined (__GCC__) // GCC
#define thread_local __thread
#endif

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <queue>
#include <stack>
#include <map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <sstream>
#if defined(unix) || defined(__unix__)
//under POSIX system, we use clock_gettime()
//remember we have to use linker option "-lrt"
#include <sys/time.h>
#include <unistd.h>
#else
#include <time.h>
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif

#include <omp.h>
#include <random>
#include <time.h>
#include <thread>
#include <set>
#include <chrono>
#include <fstream>

#include <mutex>
#include <atomic>
extern std::atomic<size_t> _G_COST;

#if defined(__GNUC__)
#include <atomic>
#include <cstring>
#include <cfloat>
#include <math.h>
inline int fopen_s(FILE** pFile, const char* path, const char* mode)
{
	if ((*pFile = fopen64(path, mode)) == NULL) return 0;
	else return 1;
}

#elif defined _MSC_VER
#else
#endif

#include "basis.h"

//Created by ZHAO, Xi at 2024.08.13. 16:39
//Integrated from my old version of HCNNG used in LSH-APG
namespace hcnnglib {

	using namespace std;
	struct knnBen
	{
		unsigned N = 0;
		unsigned num = 0;
		int** indice = nullptr;
		float** dist = nullptr;

		~knnBen() {
			delete[] indice;
			delete[] dist;
		}
	};

	inline int rand_int(const int& min, const int& max) {
		static thread_local mt19937* generator = nullptr;
		if (!generator) generator = new mt19937(clock() + std::hash<std::thread::id>()(this_thread::get_id()));
		//if (!generator) generator = new mt19937(0);
		uniform_int_distribution<int> distribution(min, max);
		return distribution(*generator);

		//static thread_local std::mt19937 rng(int(0));
		//std::uniform_int_distribution<int> ur(min, max);
		//return ur(rng);
	}

	inline int rand_int_nosafe(int min, int max) {
		return rand() % (max - min + 1) + min;
	}


	inline float rand_float(const float& min, const float& max) {
		static thread_local mt19937* generator = nullptr;
		if (!generator) generator = new mt19937(clock() + std::hash<std::thread::id>()(this_thread::get_id()));
		uniform_real_distribution<float> distribution(min, max);
		return distribution(*generator);
	}

	struct MyRandomNumberGenerator
	{
		int operator()(int limit) const
		{
			return rand_int(0, limit - 1);
		}
	};

	struct DisjointSet {
		int* parent;
		int* rank;
		DisjointSet(int N) {
			parent = new int[N];
			rank = new int[N];
			for (int i = 0; i < N; i++) {
				parent[i] = i;
				rank[i] = 0;
			}
		}
		void _union(int x, int y) {
			int xroot = parent[x];
			int yroot = parent[y];
			int xrank = rank[x];
			int yrank = rank[y];
			if (xroot == yroot)
				return;
			else if (xrank < yrank)
				parent[xroot] = yroot;
			else {
				parent[yroot] = xroot;
				if (xrank == yrank)
					rank[xroot] = rank[xroot] + 1;
			}
		}
		int find(int x) {
			if (parent[x] != x)
				parent[x] = find(parent[x]);
			return parent[x];
		}

		~DisjointSet() {
			delete[] parent;
			delete[] rank;
		}
	};

	struct Edge {
		int v1, v2;
		float weight;
		Edge() {
			v1 = -1;
			v2 = -1;
			weight = -1;
		}
		Edge(int _v1, int _v2, float _weight) {
			v1 = _v1;
			v2 = _v2;
			weight = _weight;
		}
		bool operator<(const Edge& e) const {
			return weight < e.weight;
		}
		~Edge() { }
	};

	template <class T>
	struct Matrix {
		T* M;
		int rows, cols;
		Matrix() {
			rows = -1;
			cols = -1;
			M = NULL;
		}
		Matrix(int _rows, int _cols) {
			rows = _rows;
			cols = _cols;
			M = new T[((long int)rows) * cols];
		}

		T* operator[](int i) {
			return M + (((long int)i) * cols);
		}
		~Matrix() {
			//if(M != NULL)
			// 	delete[] M;
			M = NULL;
		}

		Matrix(string path_file, int& num, int& dim) {
			Matrix<float> points = read_fvecs(path_file, num, dim);
			M = (T*)points.M;
			rows = points.rows;
			cols = points.cols;
		}

		inline Matrix<float> read_fvecs(string path_file, int& num, int& dim) {
			std::ifstream in(path_file.c_str(), std::ios::binary);
			if (!in.is_open()) {
				std::cout << "open file error" << std::endl;
				exit(-1);
			}
			in.seekg(0, std::ios::beg);
			int cnt = 0;
			std::vector<float> buf;
			buf.reserve(5000);
			int* bufint = new int;
			float* buffloat = new float;
			in.read((char*)bufint, sizeof(int));
			if (*bufint != 0) {
				std::cout << "File format error" << std::endl;
				exit(-1);
			}

			while (1) {
				in.read((char*)buffloat, sizeof(float));
				if (*((int*)buffloat) == 1) break;

				buf.push_back(*buffloat);
			}
			dim = buf.size();
			in.seekg(0, std::ios::beg);
			in.seekg(0, std::ios::end);
			std::ios::pos_type ss = in.tellg();
			size_t fsize = (size_t)ss;
			num = (unsigned)(fsize / (dim + 1) / 4);

			//data = new float[num * dim * sizeof(float)];
			Matrix<float> data(num, dim);
			float* data_ = data.M;
			in.seekg(0, std::ios::beg);
			for (size_t i = 0; i < num; i++) {
				in.read((char*)bufint, sizeof(int));
				if (*bufint != i) {
					std::cout << "File format error" << std::endl;
					exit(-1);
				}
				in.read((char*)(data_ + i * dim), dim * sizeof(float));
			}
			in.close();

			return data;
		}

		inline void write_fvecs(string path_file, Matrix<float>& M) {
			FILE* F;
			int N;
			F = fopen(path_file.c_str(), "wb");
			N = M.rows;
			for (int i = 0; i < N; i++) {
				int Dim = M.cols;
				fwrite(&Dim, sizeof(int), 1, F);
				float* aux = new float[Dim];
				for (int j = 0; j < Dim; j++)
					aux[j] = M[i][j];
				fwrite(aux, sizeof(float), Dim, F);
				delete[] aux;
			}
			fclose(F);
		}
	};

	inline float dist_L2(float* x, float* y, int n) {
		++_G_COST;
		return sqrt(cal_L2sqr(x, y, n));
		float d = 0;
		for (int i = 0; i < n; i++)
			d += (x[i] - y[i]) * (x[i] - y[i]);
		return d;
	}

#define Graph vector<vector<Edge>>
#define AdjList vector<vector< int > >
#define not_in_set(_elto,_set) (_set.find(_elto)==_set.end())
#define in_set(_elto,_set) (_set.find(_elto)!=_set.end())
#define not_in_edges(a, b, t, noedges) (noedges.find(make_pair(min(a, b),max(a, b)))==noedges.end() || noedges[make_pair(min(a, b),max(a, b))]&(1LL<<t))

	template<typename SomeType>
	float mean_v(vector<SomeType> a) {
		float s = 0;
		for (float x : a) s += x;
		return s / a.size();
	}

	template<typename SomeType>
	float sum_v(vector<SomeType> a) {
		float s = 0;
		for (float x : a) s += x;
		return s;
	}

	template<typename SomeType>
	float max_v(vector<SomeType> a) {
		float mx = a[0];
		for (float x : a) mx = max(mx, x);
		return mx;
	}

	template<typename SomeType>
	float min_v(vector<SomeType> a) {
		float mn = a[0];
		for (float x : a) mn = min(mn, x);
		return mn;
	}

	template<typename SomeType>
	float std_v(vector<SomeType> a) {
		float m = mean_v(a), s = 0, n = a.size();
		for (float x : a) s += (x - m) * (x - m);
		return sqrt(s / (n - 1));
	}

	template<typename SomeType>
	float variance_v(vector<SomeType> a, float mean) {
		float o = 0;
		for (float x : a) o += (x - mean) * (x - mean);
		return o / a.size();
	}

	inline float get_recall(vector<int> r1, vector<int> r2, int K) {
		set<int> a(r1.begin(), r1.begin() + K);
		set<int> b(r2.begin(), r2.begin() + K);
		set<int> result;
		set_intersection(a.begin(), a.end(), b.begin(), b.end(), inserter(result, result.begin()));
		return (float)result.size() / a.size();
	}

	inline float get_recall_dist(vector<float> r1, vector<float> r2) {
		sort(r1.begin(), r1.end());
		sort(r2.begin(), r2.end());
		int count = 0;
		for (int i = 0, j = 0; i < r1.size() && j < r2.size(); ) {
			if (r1[i] == r2[j]) {
				count++;
				i++;
				j++;
			}
			else if (r1[i] < r2[j])
				i++;
			else
				j++;
		}
		return (float)count / (float)r1.size();
	}

	inline long int fsize(FILE* fp) {
		long int prev = ftell(fp);
		fseek(fp, 0L, SEEK_END);
		long int sz = ftell(fp);
		fseek(fp, prev, SEEK_SET);
		return sz;
	}

	class hcnng {
		int xxx = 0;
		size_t report_every = 0;
		size_t next_report = 0;
		lsh::progress_display* pd = nullptr;
	public:
		hcnng(std::string& datasetName, Matrix<float>& points, std::string& file_graph, std::string& index_result,
			int num_cl, int minsize_cl, int max_mst_degree, bool rebuilt) {
			FILE* fp = nullptr;
			//Matrix<float> points = read_fvecs(file_dataset, N, Dim);
			int Dim = points.cols;
			int N = points.rows;
			if (rebuilt || !findIndex(file_graph)) {
				_G_COST = 0;
				auto s = std::chrono::high_resolution_clock::now();
				Graph nngraph = HCNNG_create_graph(points, Dim, num_cl, minsize_cl, max_mst_degree);
				auto e = std::chrono::high_resolution_clock::now();
				write_graph(file_graph, nngraph);
				std::chrono::duration<double> diff = e - s;
				float query_time = diff.count();
				float cc = (float)_G_COST / N;
				_G_COST = 0;

				printf("%s:\nIndexingTime=%f s, cc=%f.\n\n", datasetName.c_str(), query_time, cc);
				fopen_s(&fp, index_result.c_str(), "a");
				if (fp) fprintf(fp, "%s:\nIndexingTime=%f s, cc=%f.\n\n", datasetName.c_str(), query_time, cc);
				fclose(fp);
			}
		}

		void search(Matrix<float> queries, std::string& file_gt, std::string& file_graph, Matrix<float>& points, std::string& index_result) {
			FILE* fp = nullptr;
			//vector<vector<int> > gt = read_ivecs(file_gt, num_queries, nn_gt);
			knnBen gt;
			ben_load(gt, &(file_gt[0]));
			printf("groundtruth read...\n");

			AdjList graph = read_adjlist(file_graph, points, false);
			fopen_s(&fp, index_result.c_str(), "a");
			if (fp) print_stats_graph(fp, graph);
			fclose(fp);

			vector<vector<int>> res;
			int K = 50;
			int ef = 50;
			showInfo(points, gt, graph);

			//if (ef < K) continue;
			size_t costTotal = 0;
			int ef1 = ef + 100;
			res.clear();
			auto s = std::chrono::high_resolution_clock::now();
			size_t cost1 = _G_COST;
			res = run_on_testset(queries, K, points, graph, ef, costTotal);
			cost1 = _G_COST - cost1;
			auto e = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> diff = e - s;
			float query_time = diff.count() / 200 * 1e3;
			float recall = 0.0f, ratio = 0.0f;
			float cost = (float)costTotal / 200;
			//cost = (float)cost1 / ((size_t)num_queries);
			test_recall(gt, res, recall, ratio, points[0], queries[0], queries.cols, K);

			std::stringstream ss;
			ss << std::setw(8) << ef1
				<< std::setw(8) << K
				<< std::setw(12) << query_time
				<< std::setw(12) << recall
				<< std::setw(12) << ratio
				<< std::setw(12) << cost
				//<< std::setw(_lspace) << hop
				<< std::endl;

			cout << ss.str();

		}

		inline void sort_edges(Graph& G) {
			int N = G.size();
#pragma omp parallel for
			for (int i = 0; i < N; i++)
				sort(G[i].begin(), G[i].end());
		}

		inline Graph complete_edges(Graph& adjlist) {
			int N = adjlist.size();
			set<pair<int, int> > all;
			Graph new_graph(N);
			for (int i = 0; i < N; i++) {
				for (Edge& e : adjlist[i]) {
					if (not_in_set(make_pair(e.v1, e.v2), all) && not_in_set(make_pair(e.v2, e.v1), all)) {
						new_graph[e.v1].push_back(Edge(e.v1, e.v2, e.weight));
						new_graph[e.v2].push_back(Edge(e.v2, e.v1, e.weight));
						all.insert(make_pair(e.v1, e.v2));
						all.insert(make_pair(e.v2, e.v1));
					}

				}
			}
			return new_graph;
		}

		/*inline vector<vector<int> > read_ivecs(string path_file, int& N, int& Dim) {
			vector<vector<int> > data;
			FILE* F;
			F = fopen(path_file.c_str(), "rb");
			if (F == NULL) {
				printf("Dataset not found\n");
				exit(0);
			}
			int xxx = fread(&Dim, sizeof(int), 1, F);
			long int sizebytes = fsize(F);
			N = sizebytes / (sizeof(int) * (Dim + 1));
			rewind(F);
			for (int i = 0; i < N; i++) {
				xxx = fread(&Dim, sizeof(int), 1, F);
				int* nn = new int[Dim];
				xxx = fread(nn, sizeof(int), Dim, F);
				data.push_back(vector<int>(nn, nn + Dim));
				delete[] nn;
			}
			fclose(F);
			return data;
		}

		inline void write_ivecs(string path_file, vector<vector<int>>& GT) {
			FILE* F;
			int N;
			F = fopen(path_file.c_str(), "wb");
			N = GT.size();
			for (int i = 0; i < N; i++) {
				int K = GT[i].size();
				fwrite(&K, sizeof(int), 1, F);
				int* aux = new int[K];
				for (int j = 0; j < K; j++)
					aux[j] = GT[i][j];
				fwrite(aux, sizeof(int), K, F);
				delete[] aux;
			}
			fclose(F);
		}*/

		inline vector<int> get_sizeadj(Graph& G) {
			vector<int> NE(G.size());
			for (int i = 0; i < G.size(); i++)
				NE[i] = G[i].size();
			return NE;
		}

		inline vector<int> get_sizeadj(AdjList& G) {
			vector<int> NE(G.size());
			for (int i = 0; i < G.size(); i++)
				NE[i] = G[i].size();
			return NE;
		}

		inline void print_stats_graph(Graph& G) {
			vector<int> sizeadj;
			sizeadj = get_sizeadj(G);
			printf("\n***************************\n");
			printf("num edges:\t%.0lf\n", sum_v(sizeadj) / 2);
			printf("max degree:\t%.0lf\n", max_v(sizeadj));
			printf("min degree:\t%.0lf\n", min_v(sizeadj));
			printf("avg degree:\t%.2lf\n", mean_v(sizeadj));
			printf("std degree:\t%.2lf\n\n", std_v(sizeadj));
			printf("***************************\n\n");
		}

		inline void print_stats_graph(FILE* fp, AdjList& G) {
			vector<int> sizeadj;
			sizeadj = get_sizeadj(G);
			printf("\n***************************\n");
			printf("num edges:\t%.0lf\n", sum_v(sizeadj) / 2);
			printf("max degree:\t%.0lf\n", max_v(sizeadj));
			printf("min degree:\t%.0lf\n", min_v(sizeadj));
			printf("avg degree:\t%.2lf\n", mean_v(sizeadj));
			printf("std degree:\t%.2lf\n\n", std_v(sizeadj));
			printf("***************************\n\n");

			fprintf(fp, "\n***************************\n");
			fprintf(fp, "num edges:\t%.0lf\n", sum_v(sizeadj) / 2);
			fprintf(fp, "max degree:\t%.0lf\n", max_v(sizeadj));
			fprintf(fp, "min degree:\t%.0lf\n", min_v(sizeadj));
			fprintf(fp, "avg degree:\t%.2lf\n", mean_v(sizeadj));
			fprintf(fp, "std degree:\t%.2lf\n\n", std_v(sizeadj));
			fprintf(fp, "***************************\n\n");
		}

		inline void write_graph(string path_file, Graph& G) {
			FILE* F;
			int N;
			//F = fopen(path_file.c_str(), "wb");
			fopen_s(&F, path_file.c_str(), "wb");
			N = G.size();
			for (int i = 0; i < N; i++) {
				int degree = G[i].size();
				fwrite(&degree, sizeof(int), 1, F);
				int* aux = new int[degree];
				for (int j = 0; j < degree; j++)
					aux[j] = G[i][j].v2;
				fwrite(aux, sizeof(int), degree, F);
				delete[] aux;
			}
			fclose(F);
		}

		inline Graph read_graph(string path_file, Matrix<float>& points, bool verbose) {
			FILE* F;
			int N = points.rows;
			//F = fopen(path_file.c_str(), "rb");
			fopen_s(&F, path_file.c_str(), "rb");
			if (F == NULL) {
				printf("Graph not found\n");
				exit(0);
			}
			Graph G(N);
			int i = 0;
			int degree;
			while (fread(&degree, sizeof(int), 1, F) == 1) {
				int* nn = new int[degree];
				int xxx = fread(nn, sizeof(int), degree, F);
				G[i].reserve(degree);
				for (int j = 0; j < degree; j++) {
					if (i != nn[j]) {
						float d = dist_L2(points[i], points[nn[j]], points.cols);
						G[i].push_back(Edge(i, nn[j], d));
					}
				}
				delete[] nn;
				i++;
			}
			fclose(F);
			sort_edges(G);
			//if(verbose)
			//	print_stats_graph(G);
			return G;
		}

		inline AdjList read_adjlist(string path_file, Matrix<float>& points, bool verbose) {
			FILE* F;
			int N = points.rows;
			auto flag = fopen_s(&F,path_file.c_str(), "rb");
			if (flag != NULL) {
				printf("Graph not found\n");
				exit(0);
			}
			AdjList G(N);
			int i = 0;
			int degree;
			while (fread(&degree, sizeof(int), 1, F) == 1) {
				int* nn = new int[degree];
				int xxx = fread(nn, sizeof(int), degree, F);
				G[i].reserve(degree);
				for (int j = 0; j < degree; j++) {
					if (i != nn[j]) {
						G[i].push_back(nn[j]);
					}
				}
				delete[] nn;
				i++;
			}
			fclose(F);
			return G;
		}

		inline void showInfo(Matrix<float>& points, knnBen& benchmark, vector<vector<int>>& g)
		{
			int N = g.size();
			float res = 0.0f;
			size_t sqrMat = 0;
			size_t cnt = 0, rec = 0, cnt1 = 0;
			int f = 1;
			for (int u = 0; u < g.size(); ++u) {
				cnt1 += g[u].size();
				sqrMat += g[u].size() * g[u].size();

				if (u >= benchmark.N - 200) continue;
				cnt += g[u].size();
				for (int pos = 0; pos != g[u].size(); ++pos) {
					float dist = dist_L2(points[u], points[g[u][pos]], points.cols);
					res += sqrt(dist);
				}
			}

			float ratio = 0.0f;
			for (int u = 0; u < benchmark.N - 200; ++u) {
				std::set<unsigned> set1, set2;
				std::vector<unsigned> set_intersection;
				set_intersection.clear();
				set1.clear();
				set2.clear();

				int j = 0;
				for (int pos = 0; pos != g[u].size(); ++pos) {
					set1.insert(g[u][pos]);
					set2.insert((unsigned)benchmark.indice[u + 200][j]);
					++j;
				}
				std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
					std::back_inserter(set_intersection));

				rec += set_intersection.size();
			}

			float derivation = (float)sqrMat / N - ((float)cnt1 / N) * ((float)cnt1 / N);
			derivation = sqrt(derivation);
			printf("dist=%f, cnt=%f, unique=%d, Dcnt=%f, Recall=%f\n\n", res / cnt, (float)cnt1 / N, (int)f, derivation, (float)rec / cnt);
		}

		tuple<vector<int>, vector<float>> search_KNN(float* query, int K, AdjList& graph, Matrix<float>& points, int start, int& ef) {
			int N = points.rows;
			int cost = 0;
			unordered_set<int> visited; visited.insert(start);
			priority_queue<tuple<float, int> > q, knn;
			float furthest_dist = dist_L2(points[start], query, points.cols);
			cost++;
			q.push(make_tuple(-furthest_dist, start));
			knn.push(make_tuple(furthest_dist, start));
			while (!q.empty()) {
				float d; int v;
				tie(d, v) = q.top();
				//auto& maxD = knn.top();
				if (-d > std::get<0>(knn.top())) break;
				q.pop();
				for (int u : graph[v]) {
					//if (calc_left <= 0) break;
					if (in_set(u, visited))
						continue;
					visited.insert(u);
					d = dist_L2(points[u], query, points.cols);
					cost++;
					q.push(make_tuple(-d, u));
					knn.push(make_tuple(d, u));
					if (knn.size() > ef)
						knn.pop();
				}
			}
			ef = cost;
			vector<int> nearests;
			vector<float> dists;

			while (knn.size() > K) knn.pop();

			while (!knn.empty()) {
				float x; int y;
				tie(x, y) = knn.top();
				nearests.push_back(y);
				dists.push_back(x);
				knn.pop();
			}
			reverse(nearests.begin(), nearests.end());
			reverse(dists.begin(), dists.end());
			return make_tuple(nearests, dists);
		}

		vector<vector<int>> run_on_testset(Matrix<float>& queries, int K, Matrix<float>& points, AdjList& graph, int& max_calc, size_t& cost) {
			float recall = 0;
			int N = points.rows;
			int num_queries = queries.rows;
			vector<vector<int>> res;
			int maxC = max_calc;
			//#pragma omp parallel for
			for (int i = 0; i < num_queries; i++) {
				int start = rand_int(0, N - 1);
				auto knn = search_KNN(queries[i], K, graph, points, start, max_calc);
				res.push_back(get<0>(knn));
				cost += max_calc;
				max_calc = maxC;
			}
			return res;
			//printf("Recall@%d(%d):\t%.2lf\n", K, max_calc, recall * 100 / num_queries);
		}




		void ben_load(knnBen& benchmark, char* ben_file)
		{
			std::ifstream in(ben_file, std::ios::binary);
			if (!in.good()) {
				std::cout << "Loading bench fail" << std::endl;
				exit(-1);
			}

			in.read((char*)&benchmark.N, sizeof(unsigned));
			in.read((char*)&benchmark.num, sizeof(unsigned));

			benchmark.indice = new int* [benchmark.N];
			benchmark.dist = new float* [benchmark.N];
			for (unsigned j = 0; j < benchmark.N; j++) {
				benchmark.indice[j] = new int[benchmark.num];
				in.read((char*)&benchmark.indice[j][0], sizeof(int) * benchmark.num);
			}

			for (unsigned j = 0; j < benchmark.N; j++) {
				benchmark.dist[j] = new float[benchmark.num];
				in.read((char*)&benchmark.dist[j][0], sizeof(float) * benchmark.num);
			}
			in.close();
		}

		void test_recall(knnBen& benchmark, std::vector<std::vector<int>>& res, float& recall, float& ratio, float* data_, float* query, int dim, int k)
		{
			int qn = res.size();
			recall = 0.0f;
			ratio = 0.0f;
			//std::cout << "here? begin comp recall" << std::endl;
			//efanna2e::Distance* distance_ = new efanna2e::DistanceL2();
			for (int i = 0; i < qn; ++i) {
				if (res[i].size() != k) {
					std::cout << "Returned result Error" << std::endl;
					exit(-1);
				}
				//res[i].resize(k);
				std::set<int> g, r;
				for (int j = 0; j < k; ++j) {
					g.insert(benchmark.indice[i][j]);
					r.insert(res[i][j]);
					float dist = dist_L2(data_ + (size_t)dim * (size_t)res[i][j], query + dim * i, (unsigned)dim);
					if (benchmark.dist[i][j] == 0) {
						//std::cout << "gt result Error" << std::endl;
						//exit(-1);

						ratio += 1.0f;
					}
					else ratio += sqrt(dist) / benchmark.dist[i][j];


				}
				std::vector<int> rec;
				std::set_intersection(g.begin(), g.end(), r.begin(), r.end(),
					std::back_inserter(rec));
				recall += (float)(rec.size());
			}

			recall /= qn * k;
			ratio /= qn * k;
		}

		tuple<Graph, float> kruskal(vector<Edge>& edges, int N, Matrix<float>& points, int max_mst_degree) {
			sort(edges.begin(), edges.end());
			Graph MST(N);
			DisjointSet* disjset = new DisjointSet(N);
			float cost = 0;
			for (Edge& e : edges) {
				if (disjset->find(e.v1) != disjset->find(e.v2) && MST[e.v1].size() < max_mst_degree && MST[e.v2].size() < max_mst_degree) {
					MST[e.v1].push_back(e);
					MST[e.v2].push_back(Edge(e.v2, e.v1, e.weight));
					disjset->_union(e.v1, e.v2);
					cost += e.weight;

				}
			}
			delete disjset;
			return make_tuple(MST, cost);
		}

		Graph create_exact_mst(Matrix<float>& points, int* idx_points, int left, int right, int max_mst_degree) {
			int N = right - left + 1;
			if (N == 1) {
				xxx++;
				printf("%d\n", xxx);
			}
			float cost;
			vector<Edge> full;
			Graph mst;
			full.reserve(N * (N - 1));
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++)
					if (i != j)
						full.push_back(Edge(i, j, dist_L2(points[idx_points[left + i]], points[idx_points[left + j]], points.cols)));
			}
			tie(mst, cost) = kruskal(full, N, points, max_mst_degree);
			return mst;
		}

		bool check_in_neighbors(int u, vector<Edge>& neigh) {
			for (int i = 0; i < neigh.size(); i++)
				if (neigh[i].v2 == u)
					return true;
			return false;
		}

		void create_clusters(Matrix<float>& points, int* idx_points, int left, int right, Graph& graph, int minsize_cl, vector<omp_lock_t>& locks, int max_mst_degree) {
			int num_points = right - left + 1;

			if (num_points < minsize_cl) {
				Graph mst = create_exact_mst(points, idx_points, left, right, max_mst_degree);
				for (int i = 0; i < num_points; i++) {
					for (int j = 0; j < mst[i].size(); j++) {
						omp_set_lock(&locks[idx_points[left + i]]);
						if (!check_in_neighbors(idx_points[left + mst[i][j].v2], graph[idx_points[left + i]]))
							graph[idx_points[left + i]].push_back(Edge(idx_points[left + i], idx_points[left + mst[i][j].v2], mst[i][j].weight));
						omp_unset_lock(&locks[idx_points[left + i]]);
					}
				}
				(*pd) += num_points;
			}
			else {
				int x = rand_int(left, right);
				int y = rand_int(left, right);
				while (y == x) y = rand_int(left, right);

				//x = left;
				//y = right;

				vector<pair<float, int> > dx(num_points);
				vector<pair<float, int> > dy(num_points);
				unordered_set<int> taken;
				for (int i = 0; i < num_points; i++) {
					dx[i] = make_pair(dist_L2(points[idx_points[x]], points[idx_points[left + i]], points.cols), idx_points[left + i]);
					dy[i] = make_pair(dist_L2(points[idx_points[y]], points[idx_points[left + i]], points.cols), idx_points[left + i]);
				}
				sort(dx.begin(), dx.end());
				sort(dy.begin(), dy.end());
				int i = 0, j = 0, turn = rand_int(0, 1), p = left, q = right;

				//turn = 0;

				while (i < num_points || j < num_points) {
					if (turn == 0) {
						if (i < num_points) {
							if (not_in_set(dx[i].second, taken)) {
								idx_points[p] = dx[i].second;
								taken.insert(dx[i].second);
								p++;
								turn = (turn + 1) % 2;
							}
							i++;
						}
						else {
							turn = (turn + 1) % 2;
						}
					}
					else {
						if (j < num_points) {
							if (not_in_set(dy[j].second, taken)) {
								idx_points[q] = dy[j].second;
								taken.insert(dy[j].second);
								q--;
								turn = (turn + 1) % 2;
							}
							j++;
						}
						else {
							turn = (turn + 1) % 2;
						}
					}
				}

				dx.clear();
				dy.clear();
				taken.clear();
				vector<pair<float, int> >().swap(dx);
				vector<pair<float, int> >().swap(dy);

				create_clusters(points, idx_points, left, p - 1, graph, minsize_cl, locks, max_mst_degree);
				create_clusters(points, idx_points, p, right, graph, minsize_cl, locks, max_mst_degree);
			}
		}

		using pii = pair<int, int>;

		void create_LC(Matrix<float>& points, int* idx_points, int left, int right, Graph& graph, int minsize_cl, vector<omp_lock_t>& locks, vector<pii>& pairs) {
			int num_points = right - left + 1;

			if (num_points < minsize_cl) {
				pairs.push_back(make_pair(left, right));
			}
			else {
				int x = rand_int(left, right);
				int y = rand_int(left, right);
				while (y == x) y = rand_int(left, right);

				//x = left;
				//y = right;

				vector<pair<float, int> > dx(num_points);
				vector<pair<float, int> > dy(num_points);
				unordered_set<int> taken;
				for (int i = 0; i < num_points; i++) {
					dx[i] = make_pair(dist_L2(points[idx_points[x]], points[idx_points[left + i]], points.cols), idx_points[left + i]);
					dy[i] = make_pair(dist_L2(points[idx_points[y]], points[idx_points[left + i]], points.cols), idx_points[left + i]);
				}
				sort(dx.begin(), dx.end());
				sort(dy.begin(), dy.end());
				int i = 0, j = 0, turn = rand_int(0, 1), p = left, q = right;

				//turn = 0;

				while (i < num_points || j < num_points) {
					if (turn == 0) {
						if (i < num_points) {
							if (not_in_set(dx[i].second, taken)) {
								idx_points[p] = dx[i].second;
								taken.insert(dx[i].second);
								p++;
								turn = (turn + 1) % 2;
							}
							i++;
						}
						else {
							turn = (turn + 1) % 2;
						}
					}
					else {
						if (j < num_points) {
							if (not_in_set(dy[j].second, taken)) {
								idx_points[q] = dy[j].second;
								taken.insert(dy[j].second);
								q--;
								turn = (turn + 1) % 2;
							}
							j++;
						}
						else {
							turn = (turn + 1) % 2;
						}
					}
				}

				dx.clear();
				dy.clear();
				taken.clear();
				vector<pair<float, int> >().swap(dx);
				vector<pair<float, int> >().swap(dy);

				create_LC(points, idx_points, left, p - 1, graph, minsize_cl, locks, pairs);
				create_LC(points, idx_points, p, right, graph, minsize_cl, locks, pairs);
			}
		}

		Graph HCNNG_create_graph(Matrix<float>& points, int Dim, int num_cl, int minsize_cl, int max_mst_degree) {
			size_t N = points.rows;
			size_t estimatedCC = 2 * ceil(log(N / minsize_cl)) + N * (minsize_cl - 1);
			report_every = estimatedCC / 50;
			next_report += report_every;
			Graph G(N);
			vector<omp_lock_t> locks(N);
			for (int i = 0; i < N; i++) {
				omp_init_lock(&locks[i]);
				G[i].reserve(max_mst_degree * num_cl);
			}
			printf("creating clusters...\n");

			vector<vector<pii>> tps(num_cl);
			using pr_lr = pair<int*, pii>;
			vector<pr_lr> partis;
			vector<int*> ids(num_cl, nullptr);
			int mlc = N / (256 / num_cl);
			if (mlc < minsize_cl) mlc = minsize_cl;
#pragma omp parallel for
			for (int i = 0; i < num_cl; i++) {
				int* idx_points = new int[N];
				for (int j = 0; j < N; j++)
					idx_points[j] = j;

				create_LC(points, idx_points, 0, N - 1, G, mlc, locks, tps[i]);
				printf("end BIG cluster %d\n", i);
				//delete[] idx_points;
				ids[i] = idx_points;

			}

			for (int i = 0; i < num_cl; i++) {
				int* idx_points = ids[i];
				//#pragma omp parallel for
				for (int j = 0; j < tps[i].size(); ++j) {
					auto& x = tps[i][j];
					partis.emplace_back(idx_points, x);
				}
			}

			printf("\n\nBuilding...\n");
			pd = new lsh::progress_display((size_t)N * (size_t)num_cl);
#pragma omp parallel for
			for (int i = 0; i < partis.size(); ++i) {
				auto& prs = partis[i];
				int* idx_points = prs.first;
				pii& x = prs.second;
				int left = x.first;
				int right = x.second;
				create_clusters(points, idx_points, left, right, G, minsize_cl, locks, max_mst_degree);
				//pd += right - left + 1;
			}

			for (int i = 0; i < num_cl; i++) {
				int* idx_points = ids[i];
				delete[] idx_points;
			}
			printf("sorting...\n");
			sort_edges(G);
			print_stats_graph(G);
			return G;
		}

//		Graph HCNNG_create_graph0(Matrix<float>& points, int Dim, int num_cl, int minsize_cl, int max_mst_degree) {
//			size_t N = points.rows;
//			size_t estimatedCC = 2 * ceil(log(N / minsize_cl)) + N * (minsize_cl - 1);
//			report_every = estimatedCC / 50;
//			next_report += report_every;
//			Graph G(N);
//			vector<omp_lock_t> locks(N);
//			for (int i = 0; i < N; i++) {
//				omp_init_lock(&locks[i]);
//				G[i].reserve(max_mst_degree * num_cl);
//			}
//			printf("creating clusters...\n");
//
//			vector<vector<pii>> tps(num_cl);
//			vector<int*> ids(num_cl, nullptr);
//			int mlc = N / (64);
//			if (mlc < minsize_cl) mlc = minsize_cl;
//#pragma omp parallel for
//			for (int i = 0; i < num_cl; i++) {
//				int* idx_points = new int[N];
//				for (int j = 0; j < N; j++)
//					idx_points[j] = j;
//
//				create_LC(points, idx_points, 0, N - 1, G, mlc, locks, tps[i]);
//				printf("end BIG cluster %d\n", i);
//				//delete[] idx_points;
//				ids[i] = idx_points;
//
//			}
//
//			for (int i = 0; i < num_cl; i++) {
//				int* idx_points = ids[i];
//#pragma omp parallel for
//				for (int j = 0; j < tps[i].size(); ++j) {
//					auto& x = tps[i][j];
//					int left = x.first;
//					int right = x.second;
//					create_clusters(points, idx_points, left, right, G, minsize_cl, locks, max_mst_degree);
//				}
//
//				printf("end cluster %d\n", i);
//				delete[] idx_points;
//
//			}
//			printf("sorting...\n");
//			sort_edges(G);
//			print_stats_graph(G);
//			return G;
//		}
//
//		Graph HCNNG_create_graph1(Matrix<float>& points, int Dim, int num_cl, int minsize_cl, int max_mst_degree) {
//			size_t N = points.rows;
//			size_t estimatedCC = 2 * ceil(log(N / minsize_cl)) + N * (minsize_cl - 1);
//			report_every = estimatedCC / 50;
//			next_report += report_every;
//			Graph G(N);
//			vector<omp_lock_t> locks(N);
//			for (int i = 0; i < N; i++) {
//				omp_init_lock(&locks[i]);
//				G[i].reserve(max_mst_degree * num_cl);
//			}
//			printf("creating clusters...\n");
//#pragma omp parallel for
//			for (int i = 0; i < num_cl; i++) {
//				int* idx_points = new int[N];
//				for (int j = 0; j < N; j++)
//					idx_points[j] = j;
//
//				create_clusters(points, idx_points, 0, N - 1, G, minsize_cl, locks, max_mst_degree);
//				printf("end cluster %d\n", i);
//				delete[] idx_points;
//			}
//			printf("sorting...\n");
//			sort_edges(G);
//			print_stats_graph(G);
//			return G;
//		}

		bool findIndex(string& file)
		{
			std::ifstream in(file.c_str(), std::ios::binary);
			if (!in.is_open()) {
				return false;
			}
			in.close();
			return true;
		}
	};
}