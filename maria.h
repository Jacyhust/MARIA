#pragma once
#include "hnswlib.h"
#include <mutex>
#include <algorithm>
using hnsw = hnswlib::HierarchicalNSW<float>;
extern std::unique_lock<std::mutex>* glock;

class maria
{
private:
	std::string index_file;

public:
	int N;
	int dim;
	int S;
	int L;
	int K;

	std::string alg_name = "maria";
	Partition parti;
	Preprocess* prep = nullptr;
	IpSpace* ips = nullptr;
	hnsw** apgs = nullptr;
public:
	maria(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :parti(part_) {
		N = param_.N;
		dim = param_.dim + 1;
		L = param_.L;
		K = param_.K;
		S = param_.S;
		prep=&prep_; 
		randomXT();
		buildIndex();
	}

	void buildIndex() {
		int M = 24;
		int ef = 40;
		ips = new IpSpace(dim);
		apgs = new hnsw* [parti.numChunks];
		size_t report_every = N / 20;
		if (report_every > 1e5) report_every = 1e5;

		int j1 = 0;
		for (int i = 0; i < parti.numChunks; ++i) {
			apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i];
			auto id = parti.EachParti[i][0];
			auto data = prep->data.val[id];
			appr_alg->addPoint((void*)(data), (size_t)id);
			std::mutex inlock;

			auto vecsize = parti.nums[i];
			lsh::timer timer;
#pragma omp parallel for
			for (int k = 1; k < vecsize; k++) {
				size_t j2 = 0;
#pragma omp critical
				{
					j1++;
					j2 = j1;
					if (j1 % report_every == 0) {
						std::cout << (int)round(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
						timer.restart();
					}
				}
				j2 = parti.EachParti[i][k];
				float* data = prep->data.val[j2];
				appr_alg->addPoint((void*)(data), (size_t)j2);
			}
		}
	}

	void randomXT() {
		std::mt19937 rng(int(std::time(0)));
		std::uniform_real_distribution<float> ur(-1, 1);
		int count = 0;
		for (int j = 0; j < N; j++)
		{
			assert(parti.MaxLen[parti.chunks[j]] >= prep->SquareLen[j]);
			prep->data.val[j][dim - 1] = sqrt(parti.MaxLen[parti.chunks[j]] - prep->SquareLen[j]);
			if (ur(rng) > 0) {
				prep->data.val[j][dim - 1] *= -1;
				++count;
			}
		}
	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();
	
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && (1.0f-q->resHeap.top().dist) > 
				q->norm * sqrt(parti.MaxLen[i])) break;


			//apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i];
			auto id = parti.EachParti[i][0];
			auto data = prep->data.val[id];
			//appr_alg->addPoint((void*)(data), (size_t)id);
			//std::mutex inlock;
			appr_alg->setEf(q->k+1000);
			auto res = appr_alg->searchKnn(q->queryPoint, q->k);

			while (!res.empty()) {
				auto top = res.top();
				res.pop();
				q->resHeap.emplace(top.second, top.first);
				while (q->resHeap.size() > q->k) q->resHeap.pop();
			}

			
		}

		while (!q->resHeap.empty()) {
			auto top = q->resHeap.top();
			q->resHeap.pop();
			q->res.emplace_back(top.id, 1.0-top.dist);
		}
		
		std::reverse(q->res.begin(), q->res.end());

		q->time_total = timer.elapsed();
	}

	//void GetTables(Preprocess& prep);
	//bool IsBuilt(const std::string& file);
	~maria() {
		for (int i = 0; i < parti.numChunks; ++i) {
			delete apgs[i];
		}
		delete[] apgs;
	}
};

class myHNSW {
private:
	std::string index_file;
	IpSpace* ips = nullptr;
	hnsw* apg = nullptr;
	Preprocess* prep = nullptr;
public:
	int N;
	int dim;
	// Number of hash functions
	int S;
	//#L Tables; 
	int L;
	// Dimension of the hash table
	int K;

	std::string alg_name = "hnsw";

	myHNSW(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable){
		N = param_.N;
		dim = param_.dim;
		L = param_.L;
		K = param_.K;
		S = param_.S;
		prep = &prep_;
		//GetHash();
		buildIndex();
	}

	void setEf(size_t ef){
		apg->setEf(ef);
	}

	void buildIndex() {
		int M = 24;
		int ef = 40;
		ips = new IpSpace(dim);
		//apg = new hnsw[parti.numChunks];
		size_t report_every = N / 20;
		if (report_every > 1e5) report_every = 1e5;

		int j1 = 0;
		apg = new hnsw(ips, N, M, ef);
		auto id = 0;
		auto data = prep->data.val[id];
		apg->addPoint((void*)(data), (size_t)id);
		std::mutex inlock;

		auto vecsize = N;
		lsh::timer timer;
#pragma omp parallel for
		for (int k = 1; k < vecsize; k++) {
			size_t j2 = 0;
#pragma omp critical
			{
				j1++;
				j2 = j1;
				if (j1 % report_every == 0) {
					std::cout << (int)(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
					timer.restart();
				}
			}
			j2 = k;
			float* data = prep->data.val[j2];
			apg->addPoint((void*)(data), (size_t)j2);
		}
	}


	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();
		int ef = apg->ef_;
		//apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
		auto& appr_alg = apg;
		auto id = 0;
		auto data = prep->data.val[id];
		//appr_alg->addPoint((void*)(data), (size_t)id);
		//std::mutex inlock;
		auto res = appr_alg->searchKnn(q->queryPoint, q->k + ef);

		while (!res.empty()) {
			auto top = res.top();
			res.pop();
			q->resHeap.emplace(top.second, top.first);
			while (q->resHeap.size() > q->k) q->resHeap.pop();
		}

		while (!q->resHeap.empty()) {
			auto top = q->resHeap.top();
			q->resHeap.pop();

			q->res.emplace_back(top.id, 1.0-top.dist);
		}

		std::reverse(q->res.begin(), q->res.end());

		q->time_total = timer.elapsed();
	}

	~myHNSW() {
		delete apg;
	}
};

class mariaV2
{
private:
	std::string index_file;

public:
	int N;
	int dim;
	// Number of hash functions
	int S;
	//#L Tables; 
	int L;
	// Dimension of the hash table
	int K;
	std::vector<int> orders_in_Parti;
	Partition parti;
	Preprocess* prep = nullptr;
	IpSpace* ips = nullptr;
	hnsw** apgs = nullptr;

	std::vector<int> interEdges;
	std::string alg_name = "mariaV2";
public:
	mariaV2(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :parti(part_) {
		N = param_.N;
		dim = param_.dim + 1;
		L = param_.L;
		K = param_.K;
		S = param_.S;
		prep = &prep_;
		randomXT();
		buildIndex();
		interConnect();
	}

	void buildIndex() {
		int M = 24;
		int ef = 40;
		ips = new IpSpace(dim);
		apgs = new hnsw * [parti.numChunks];
		size_t report_every = N / 20;
		if (report_every > 1e5) report_every = 1e5;
		orders_in_Parti.resize(N);
		int j1 = 0;
		for (int i = 0; i < parti.numChunks; ++i) {
			apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i];
			auto id = parti.EachParti[i][0];
			auto data = prep->data.val[id];
			appr_alg->addPoint((void*)(data), (size_t)id);
			std::mutex inlock;

			auto vecsize = parti.nums[i];
			lsh::timer timer;
#pragma omp parallel for
			for (int k = 1; k < vecsize; k++) {
				size_t j2 = 0;
#pragma omp critical
				{
					j1++;
					j2 = j1;
					if (j1 % report_every == 0) {
						std::cout << (int)(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
						timer.restart();
					}
				}
				j2 = parti.EachParti[i][k];
				orders_in_Parti[j2] = k;
				float* data = prep->data.val[j2];
				appr_alg->addPoint((void*)(data), (size_t)j2);
			}
		}
	}



	void randomXT() {
		std::mt19937 rng(int(std::time(0)));
		std::uniform_real_distribution<float> ur(-1, 1);
		int count = 0;
		for (int j = 0; j < N; j++)
		{
			assert(parti.MaxLen[parti.chunks[j]] >= prep->SquareLen[j]);
			prep->data.val[j][dim - 1] = sqrt(parti.MaxLen[parti.chunks[j]] - prep->SquareLen[j]);
			if (ur(rng) > 0) {
				prep->data.val[j][dim - 1] *= -1;
				++count;
			}
		}
	}

	void interConnect() {
		interEdges.resize(N, 0);
		
		for (int i = 1; i < parti.numChunks; ++i) {
			//apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i - 1];
			//auto id = parti.EachParti[i][0];
			//auto data = prep->data.val[id];
			//appr_alg->addPoint((void*)(data), (size_t)id);
			//std::mutex inlock;

			auto vecsize = parti.nums[i];
			lsh::timer timer;
#pragma omp parallel for
			for (int k = 0; k < vecsize; k++) {
				auto id = parti.EachParti[i][k];
				auto data = prep->data.val[id];	
				auto res = appr_alg->searchKnnWithDist(data, 1, cal_L2sqr_hnsw);

				interEdges[id] = res.top().second;
			}
			//auto res = appr_alg->searchKnnWithDist(q->queryPoint, 1, cal_L2sqr_hnsw);
		}
	}

	void knn_simple(queryN* q) {
		lsh::timer timer;
		timer.restart();

		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && q->resHeap.top().dist > 
				q->norm * sqrt(parti.MaxLen[i])) break;

			auto& appr_alg = apgs[i];
			auto res = appr_alg->searchKnn(q->queryPoint, q->k);

			while (!res.empty()) {
				auto top = res.top();
				res.pop();
				q->resHeap.emplace(top.second, 1.0f - top.first);
				while (q->resHeap.size() > q->k) q->resHeap.pop();
			}
		}

		while (!q->resHeap.empty()) {
			auto top = q->resHeap.top();
			q->resHeap.pop();

			q->res.emplace_back(top.id, top.dist);
		}

		std::reverse(q->res.begin(), q->res.end());

		q->time_total = timer.elapsed();
	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();

		//unsigned int ep_id = parti.EachParti[parti.numChunks - 1][0];
		unsigned int ep_id = 0;
		Res nn0 = Res(-1, -FLT_MAX);

		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && q->resHeap.top().dist > 
				q->norm * sqrt(parti.MaxLen[i])) break;

			auto& appr_alg = apgs[i];
			
			if (ep_id >= appr_alg->max_elements_|| ep_id < 0) {
				std::cerr << ep_id << "  Illegal id!\n";
				exit(-1);
			}
			//auto res= appr_alg->searchKnn(q->queryPoint, q->k);
			auto res = appr_alg->searchBaseLayerST<false>(ep_id, q->queryPoint, (size_t)(q->k));
			//searchKnn
			//(q->queryPoint, q->k);

			while (!res.empty()) {
				auto top = res.top();
				if (parti.chunks[appr_alg->getExternalLabel(top.second)] != i) {
					std::cerr << "  Finding wrong points!\n";
				}
				if(res.size()==1){
					nn0=Res(appr_alg->getExternalLabel(top.second), top.first);
					// if(1.0f-top.first>nn0.dist){
					// 	nn0=
					// }
				}
				res.pop();
				
				q->resHeap.emplace(appr_alg->getExternalLabel(top.second), top.first);
				while (q->resHeap.size() > q->k) q->resHeap.pop();

				
			}

			ep_id = interEdges[nn0.id];
			ep_id = orders_in_Parti[ep_id];

			
		}

		while (!q->resHeap.empty()) {
			auto top = q->resHeap.top();
			q->resHeap.pop();

			q->res.emplace_back(top.id, 1.0-top.dist);
		}

		std::reverse(q->res.begin(), q->res.end());

		q->time_total = timer.elapsed();
	}

	//void GetTables(Preprocess& prep);
	//bool IsBuilt(const std::string& file);
	~mariaV2() {
		for (int i = 0; i < parti.numChunks; ++i) {
			delete apgs[i];
		}
		delete[] apgs;
	}
};

using hc_mips=hcnngLite::hcnng<calInnerProductReverse>;

class mariaV3
{
private:
	std::string index_file;

public:
	int N;
	int dim;
	// Number of hash functions
	int S;
	//#L Tables; 
	int L;
	// Dimension of the hash table
	int K;

	//float** hashval;
	Partition parti;
	Preprocess* prep = nullptr;
	L2Space* ips = nullptr;
	hnsw** apgs = nullptr;
	//HashParam hashpar;
	//std::vector<int>*** myIndexes;

	//float tmin;
	//float tstep;
	//float smin;
	//float sstep;
	//int rows;
	//int cols;
	//float** phi;

	//void load_funtable(const std::string& file);
	std::string alg_name = "mariaV3";
public:
	mariaV3(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :parti(part_) {
		N = param_.N;
		dim = param_.dim + 1;
		L = param_.L;
		K = param_.K;
		S = param_.S;
		prep = &prep_;
		randomXT();
		buildIndex();
	}

	void buildIndex() {
		int M = 24;
		int ef = 40;
		ips = new L2Space(dim);
		apgs = new hnsw * [parti.numChunks];
		size_t report_every = N / 20;
		if (report_every > 1e5) report_every = 1e5;

		int j1 = 0;
		for (int i = 0; i < parti.numChunks; ++i) {
			apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i];
			auto id = parti.EachParti[i][0];
			auto data = prep->data.val[id];
			appr_alg->addPoint((void*)(data), (size_t)id);
			std::mutex inlock;

			auto vecsize = parti.nums[i];
			lsh::timer timer;
#pragma omp parallel for
			for (int k = 1; k < vecsize; k++) {
				size_t j2 = 0;
#pragma omp critical
				{
					j1++;
					j2 = j1;
					if (j1 % report_every == 0) {
						std::cout << (int)round(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
						timer.restart();
					}
				}
				j2 = parti.EachParti[i][k];
				float* data = prep->data.val[j2];
				appr_alg->addPoint((void*)(data), (size_t)j2);
			}
		}
	}



	void randomXT() {
		std::mt19937 rng(int(std::time(0)));
		std::uniform_real_distribution<float> ur(-1, 1);
		int count = 0;
		for (int j = 0; j < N; j++)
		{
			assert(parti.MaxLen[parti.chunks[j]] >= prep->SquareLen[j]);
			prep->data.val[j][dim - 1] = sqrt(parti.MaxLen[parti.chunks[j]] - prep->SquareLen[j]);
			if (ur(rng) > 0) {
				prep->data.val[j][dim - 1] *= -1;
				++count;
			}
		}
	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();

		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && q->resHeap.top().dist >
				q->norm * sqrt(parti.MaxLen[i])) break;


			//apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i];
			auto id = parti.EachParti[i][0];
			auto data = prep->data.val[id];
			//appr_alg->addPoint((void*)(data), (size_t)id);
			//std::mutex inlock;
			//auto res = appr_alg->searchKnn(q->queryPoint, q->k);
			auto res = appr_alg->searchKnnWithDist(data, q->k, cal_inner_product_hnsw);

			while (!res.empty()) {
				auto top = res.top();
				res.pop();
				q->resHeap.emplace(top.second, top.first);
				while (q->resHeap.size() > q->k) q->resHeap.pop();
			}


		}

		while (!q->resHeap.empty()) {
			auto top = q->resHeap.top();
			q->resHeap.pop();
			q->res.emplace_back(top.id, top.dist);
		}

		std::reverse(q->res.begin(), q->res.end());

		q->time_total = timer.elapsed();
	}

	//void GetTables(Preprocess& prep);
	//bool IsBuilt(const std::string& file);
	~mariaV3() {
		for (int i = 0; i < parti.numChunks; ++i) {
			delete apgs[i];
		}
		delete[] apgs;
	}
};

class maria_hcnng
{
private:
	std::string index_file;

public:
	int N;
	int dim;
	// Number of hash functions
	int S;
	//#L Tables; 
	int L;
	// Dimension of the hash table
	int K;

	std::string alg_name = "maria_hc";
	//float** hashval;
	Partition parti;
	Preprocess* prep = nullptr;
	IpSpace* ips = nullptr;
	hc_mips** apgs = nullptr;
	//HashParam hashpar;
	//std::vector<int>*** myIndexes;

	//float tmin;
	//float tstep;
	//float smin;
	//float sstep;
	//int rows;
	//int cols;
	//float** phi;

	//void load_funtable(const std::string& file);
public:
	maria_hcnng(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :parti(part_) {
		N = param_.N;
		dim = param_.dim + 1;
		L = param_.L;
		K = param_.K;
		S = param_.S;
		prep=&prep_; 
		index_file=file;

		randomXT();
		buildIndex();
	}

	
	void buildIndex(){
		int minsize_cl = 500;
		int num_cl = 10;
		int max_mst_degree = 3;

		apgs = new hc_mips * [parti.numChunks];
		for (int i = 0; i < parti.numChunks; ++i) {
			Data data;
			data.N = parti.nums[i];
			data.dim = dim - 1;
			data.val = new float* [data.N];
			for (int j = 0; j < data.N; ++j) {
				data.val[j] = prep->data.val[parti.EachParti[i][j]];
			}

			apgs[i] = new hc_mips(index_file + std::to_string(i), data, "indexes/"+index_file + std::to_string(i),
				"index_result.txt", minsize_cl, num_cl, max_mst_degree, 1);
		}
	}



	void randomXT() {
		std::mt19937 rng(int(std::time(0)));
		std::uniform_real_distribution<float> ur(-1, 1);
		int count = 0;
		for (int j = 0; j < N; j++)
		{
			assert(parti.MaxLen[parti.chunks[j]] >= prep->SquareLen[j]);
			prep->data.val[j][dim - 1] = sqrt(parti.MaxLen[parti.chunks[j]] - prep->SquareLen[j]);
			if (ur(rng) > 0) {
				prep->data.val[j][dim - 1] *= -1;
				++count;
			}
		}
	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();
	
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && (-(q->resHeap.top().dist)) > 
				q->norm * sqrt(parti.MaxLen[i])) break;


			//apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i];
			auto start = 0;
			int ef = q->k + 100;
			//appr_alg->addPoint((void*)(data), (size_t)id);
			//std::mutex inlock;
			appr_alg->knn4maria(q, parti.EachParti[i], start, ef);
		}

		while (q->resHeap.size()>q->k) q->resHeap.pop();

		while (!q->resHeap.empty()) {
			auto top = q->resHeap.top();
			q->resHeap.pop();
			q->res.emplace_back(top.id, -top.dist);
		}
		
		std::reverse(q->res.begin(), q->res.end());

		q->time_total = timer.elapsed();
	}

	//void GetTables(Preprocess& prep);
	//bool IsBuilt(const std::string& file);
	~maria_hcnng() {
		for (int i = 0; i < parti.numChunks; ++i) {
			delete apgs[i];
		}
		delete[] apgs;
	}
};