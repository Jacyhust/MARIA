#pragma once
#include "hnswlib.h"
#include <mutex>
#include <algorithm>

struct clusters {
	//float* centroid = nullptr;
	int centroid_id = 0;
	std::vector<int> points;
	clusters() = default;
	clusters(int id) :centroid_id(id) {}
};

class solidAnglePartition {
	int N;
	int dim;
	int S;
	int L;
	int K;
	Preprocess* prep = nullptr;
	Partition* parti = nullptr;
	std::vector<clusters> clusts;
public:
	solidAnglePartition(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable){
		N = param_.N;
		dim = param_.dim + 1;
		L = param_.L;
		K = param_.K;
		S = param_.S;
		prep = &prep_;
		parti = &part_;
		//randomXT();
		buildIndex();
		
	}

	void buildIndex() {
		std::vector<Res> pairs(N);
		for (int i = 0; i < N; ++i) {
			pairs[i] = Res(i, prep->norms[i]);
		}

		std::sort(pairs.begin(), pairs.end(), std::greater<Res>());

		//int i = 0;
		auto& data = prep->data.val;
		for (int i = 0; i < N; ++i) {
			bool is_put = false;
			for (auto& clust: clusts) {
				if (cal_cosine_similarity(data[clust.centroid_id], data[pairs[i].id], dim,
					prep->norms[clust.centroid_id], prep->norms[pairs[i].id]) > 0.8) {
					clust.points.emplace_back(pairs[i].id);
					is_put = true;
					break;
				}
			}
			if (!is_put) {
				clusts.emplace_back(pairs[i].id);
			}
		}

		std::cout << "There are " << clusts.size() << " clusters";
	}

	void knn(queryN* q) {
		//for (auto& clust : clusts) {
		//	if()
		//}
	}
};

#include "maria.h"

class mariaV4 : public maria_hcnng {
	Data normD;
	//float** normlizedData = nullptr;
	myHNSW* hnsw = nullptr;
	std::string alg_name = "mariaV4";
	int size_per_data = 0;
	int* links = nullptr;
public:
	mariaV4(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :
		maria_hcnng(prep_, param_, file, part_, funtable) {
		normlize();
		hnsw = new myHNSW(normD, param_, file, part_, funtable);
		size_per_data = hnsw->getM();
		size_per_data += M + 1;
		links = new int[N * size_per_data];

		for (int i = 0; i < N; ++i) {
			 
		}
	}

	void normlize() {
		float** normlizedData = new float* [N];
		for (int i = 0; i < N; ++i) {
			normlizedData[i] = new float[dim];
			for (int j = 0; j < dim; ++j) {
				normlizedData[i][j] = prep->data[i][j] / prep->norms[i];
			}
		}
		normD.N = N;
		normD.dim = dim;
		normD.val = normlizedData;

	}

	~mariaV4() {
		clear_2d_array(normD.val, N);
	}
};