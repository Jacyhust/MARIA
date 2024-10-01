#pragma once
#include "StructType.h"
#include "basis.h"
#include <cmath>
#include <assert.h>
#include <string>
#include <vector>
#include <queue>
#include <cfloat>
class Preprocess
{
public:
	Data data;
	Data queries;
	float* norms = nullptr;
	Ben benchmark;
	float MaxLen;
	std::string data_file;
	std::string ben_file;
public:
	Preprocess(const std::string& path, const std::string& ben_file_);
	void load_data(const std::string& path);
	void load_fbin(const std::string& path, Data& data);
	void cal_SquareLen();
	void ben_make();
	void ben_save();
	void ben_load();
	void ben_create();
	~Preprocess();
};

struct Dist_id
{
	int id = -1;
	float dist = 0.0f;
	//Dist_id() = default;
	Dist_id(int id_, float dist_) :id(id_), dist(dist_) {}
	bool operator < (const Dist_id& rhs) {
		return dist < rhs.dist;
	}
};

class Partition
{
private:
	float ratio;
	void make_chunks_fargo(Preprocess& prep);
	void make_chunks_maria(Preprocess& prep);
public:
	int numChunks;
	std::vector<float> MaxLen;

	//The chunk where each point belongs
	//chunks[i]=j: i-th point is in j-th parti
	std::vector<int> chunks;

	//The data size of each chunks
	//nums[i]=j: i-th parti has j points
	std::vector<int> nums;
	
	//The buckets by parti;
	//EachParti[i][j]=k: k-th point is the j-th point in i-th parti
	std::vector<std::vector<int>> EachParti;

	//std::vector<Dist_id> distpairs;
	void display();

	Partition(float c_, Preprocess& prep);

	Partition(float c_, float c0_, Preprocess& prep);
	//Partition() {}
	~Partition();
};

class Parameter //N,dim,S, L, K, M, W;
{
public:
	int N;
	int dim;
	// Number of hash functions
	int S;
	//#L Tables; 
	int L;
	// Dimension of the hash table
	int K;
	//
	int MaxSize = -1;
	//
	int KeyLen = -1;

	int M = 1;

	int W = 0;

	float U;
	Parameter(Preprocess& prep, int L_, int K_, int M);
	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_);
	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_, float W_);
	Parameter(Preprocess& prep, float c_, float S0);
	Parameter(Preprocess& prep, float c0_);
	bool operator = (const Parameter& rhs);
	~Parameter();
};

struct Res//the result of knns
{
	//dist can be:
	//1. L2-distance
	//2. The opposite number of inner product
	float dist = 0.0f;
	int id = -1;
	Res() = default;
	Res(int id_, float inp_) :id(id_), dist(inp_) {}
	bool operator < (const Res& rhs) const {
		return dist < rhs.dist;
	}

	bool operator > (const Res& rhs) const {
		return dist > rhs.dist;
	}
};

class queryN
{
public:
	// the parameter "c" in "c-ANN"
	float c;
	//which chunk is accessed
	//int chunks;

	//float R_min = 4500.0f;//mnist
	//float R_min = 1.0f;
	float init_w = 1.0f;

	float* queryPoint = NULL;
	float* hashval = NULL;
	//float** myData = NULL;
	int dim = 1;

	int UB = 0;
	float minKdist = FLT_MAX;
	// Set of points sifted
	std::priority_queue<Res> resHeap;

	//std::vector<int> keys;

public:
	// k-NN
	unsigned k = 1;
	// Indice of query point in dataset. Be equal to -1 if the query point isn't in the dataset.
	unsigned qid = -1;

	float beta = 0;
	float norm = 0.0f;
	unsigned cost = 0;

	//#access;
	int maxHop = -1;
	//
	unsigned prunings = 0;
	//cost of each partition
	std::vector<int> costs;
	//
	float time_total = 0;
	//
	float timeHash = 0;
	//
	float time_sift = 0;

	float time_verify = 0;
	// query result:<indice of ANN,distance of ANN>
	std::vector<Res> res;

public:
	queryN(unsigned id, float c_, unsigned k_, Preprocess& prep, float beta_) {
		qid = id;
		c = c_;
		k = k_;
		beta = beta_;
		//myData = prep.data.val;
		dim = prep.data.dim + 1;
		queryPoint = prep.queries[id];

		norm = sqrt(cal_inner_product(queryPoint, queryPoint, dim));
		//search();
	}

	//void search();

	~queryN() { 
		delete hashval; 
		//delete queryPoint;
	}
};