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
	
	int size_per_data = 0;
	int* links = nullptr;
	std::vector<int> visited;
	
public:
	std::string alg_name = "mariaV4";
	mariaV4(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :
		maria_hcnng(prep_, param_, file, part_, funtable) {
		normlize();
		hnsw = new myHNSW(normD, param_, file, part_, funtable);
		size_per_data = hnsw->getM();
		size_per_data += M + 1;
		links = new int[N * size_per_data];
		visited.resize(N, -1);
		fillEdges();
	}

	void fillEdges() {
		hnsw->buildMap();
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			auto& apg = apgs[i];
			for (int j = 0; j < parti.EachParti[i].size(); ++j) {
				int pid = parti.EachParti[i][j];
				int* start = links + pid * size_per_data;

				hnsw->getEdgeSet(pid, start);
				//auto ptr = hnsw->getEdgeSet(pid);// the edge set in hnsw
				if (apg->nngraph[j].size() == 0) continue;
				auto& edgesInBlock = apg->nngraph[j];// the edge set in hcnng
				
				//int size = start[0]; // size of the edge set in hnsw
				
				//int len = size + edgesInBlock.size();
				//start[0] = len;

				//for (int r = 1; r < size + 1; ++r) {
				//	start[r + size + 1] = parti.EachParti[i][edgesInBlock[r].id];
				//}

				//memcpy(start + 1, ptr + 1, size);
				//int r = 1;

				for (int r = 0; r < edgesInBlock.size(); ++r) {
					start[r + start[0] + 1] = parti.EachParti[i][edgesInBlock[r].id];
				}
				start[0] += edgesInBlock.size();
			}
		}
		//for (int i = 0; i < N; ++i) {
		//	auto ptr = hnsw->getEdgeSet(i);
		//	int size = *((unsigned short int*)ptr);
		//	int len = size +
		//}
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
		//normD.val = normlizedData;
		normD.val=prep->data.val;
	}

	void knn(queryN* q) {
		int cost = 0;
		lsh::timer timer;
		std::priority_queue<Res> accessed_candidates, top_candidates;
		int start = 0;
		visited[start] = q->qid;
		int ef = q->k + 100;

		auto& data = prep->data;

		float dist = calInnerProductReverse(q->queryPoint, data[start], data.dim);
		cost++;
		accessed_candidates.emplace(start, -dist);
		top_candidates.emplace(start, dist);

		while (!accessed_candidates.empty()) {
			Res top = accessed_candidates.top();
			if (-top.dist > top_candidates.top().dist) break;
			accessed_candidates.pop();

			auto ptr = links + top.id * size_per_data;
			for (int i = 1; i < ptr[0] + 1; ++i) {
				if (visited[ptr[i]] == q->qid) continue;
				visited[ptr[i]] = q->qid;
				dist = calInnerProductReverse(q->queryPoint, data[ptr[i]], data.dim);
				cost++;
				accessed_candidates.emplace(ptr[i], -dist);
				top_candidates.emplace(ptr[i], dist);
				if (top_candidates.size() > ef) top_candidates.pop();
			}

			//for (auto& u : nngraph[top.id]) {
			//	if (visited[u.id] == q->qid) continue;
			//	visited[u.id] = q->qid;
			//	dist = dist_t(q->queryPoint, data[u.id], data.dim);
			//	cost++;
			//	accessed_candidates.emplace(u.id, -dist);
			//	top_candidates.emplace(u.id, dist);
			//	if (top_candidates.size() > ef) top_candidates.pop();
			//}
		}

		while (top_candidates.size() > q->k) top_candidates.pop();

		q->res.resize(q->k);
		int pos = q->k;
		while (!top_candidates.empty()) {
			q->res[--pos] = top_candidates.top();
			top_candidates.pop();
		}
		q->time_total = timer.elapsed();
	}

	~mariaV4() {
		clear_2d_array(normD.val, N);
	}


};

class mariaV5 : public maria_hcnng {
	Data normD;
	//float** normlizedData = nullptr;
	myHNSW* hnsw = nullptr;

	int size_per_data = 0;
	int* links = nullptr;
	std::vector<int> visited;

public:
	std::string alg_name = "mariaV5";
	mariaV5(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :
		maria_hcnng(prep_, param_, file, part_, funtable) {
		normlize();
		hnsw = new myHNSW(normD, param_, file, part_, funtable);
		size_per_data = hnsw->getM();
		size_per_data += M + 1;
		links = new int[N * size_per_data];
		visited.resize(N, -1);
		fillEdges();
	}

	void fillEdges() {
		hnsw->buildMap();
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			auto& apg = apgs[i];
			for (int j = 0; j < parti.EachParti[i].size(); ++j) {
				int pid = parti.EachParti[i][j];
				int* start = links + pid * size_per_data;

				hnsw->getEdgeSet(pid, start);
				//auto ptr = hnsw->getEdgeSet(pid);// the edge set in hnsw
				if (apg->nngraph[j].size() == 0) continue;
				auto& edgesInBlock = apg->nngraph[j];// the edge set in hcnng

				//int size = start[0]; // size of the edge set in hnsw

				//int len = size + edgesInBlock.size();
				//start[0] = len;

				//for (int r = 1; r < size + 1; ++r) {
				//	start[r + size + 1] = parti.EachParti[i][edgesInBlock[r].id];
				//}

				//memcpy(start + 1, ptr + 1, size);
				//int r = 1;

				for (int r = 0; r < edgesInBlock.size(); ++r) {
					start[r + start[0] + 1] = parti.EachParti[i][edgesInBlock[r].id];
				}
				start[0] += edgesInBlock.size();
			}
		}
		//for (int i = 0; i < N; ++i) {
		//	auto ptr = hnsw->getEdgeSet(i);
		//	int size = *((unsigned short int*)ptr);
		//	int len = size +
		//}
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
		//normD.val = normlizedData;
		normD.val = prep->data.val;
	}

	void knn(queryN* q) {
		int cost = 0;
		lsh::timer timer;
		std::priority_queue<Res> accessed_candidates, top_candidates;
		int start = 0;
		visited[start] = q->qid;
		int ef = q->k + 100;

		auto& data = prep->data;

		float dist = calInnerProductReverse(q->queryPoint, data[start], data.dim);
		cost++;
		accessed_candidates.emplace(start, -dist);
		top_candidates.emplace(start, dist);

		while (!accessed_candidates.empty()) {
			Res top = accessed_candidates.top();
			if (-top.dist > top_candidates.top().dist) break;
			accessed_candidates.pop();

			auto ptr = links + top.id * size_per_data;
			for (int i = 1; i < ptr[0] + 1; ++i) {
				if (visited[ptr[i]] == q->qid) continue;
				visited[ptr[i]] = q->qid;
				dist = calInnerProductReverse(q->queryPoint, data[ptr[i]], data.dim);
				cost++;
				accessed_candidates.emplace(ptr[i], -dist);
				top_candidates.emplace(ptr[i], dist);
				if (top_candidates.size() > ef) top_candidates.pop();
			}

			//for (auto& u : nngraph[top.id]) {
			//	if (visited[u.id] == q->qid) continue;
			//	visited[u.id] = q->qid;
			//	dist = dist_t(q->queryPoint, data[u.id], data.dim);
			//	cost++;
			//	accessed_candidates.emplace(u.id, -dist);
			//	top_candidates.emplace(u.id, dist);
			//	if (top_candidates.size() > ef) top_candidates.pop();
			//}
		}

		while (top_candidates.size() > q->k) top_candidates.pop();

		q->res.resize(q->k);
		int pos = q->k;
		while (!top_candidates.empty()) {
			q->res[--pos] = top_candidates.top();
			top_candidates.pop();
		}
		q->time_total = timer.elapsed();
	}

	~mariaV5() {
		clear_2d_array(normD.val, N);
	}


};