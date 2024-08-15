// Fargo Project: Revised by Xi ZHAO -- Nov 16, 2022

// For PVDLB 2023: FARGO: Fast Maximum Inner Product Search via Global Multi-Probing

// For any question, please feel free to contact me. Email: xzhaoca@connect.ust.hk

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <cstring>
#include <chrono>

#include "Preprocess.h"
#include "mf_alsh.h"
#include "basis.h"
#include "alg.h"
#include "maria.h"

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;
std::unique_lock<std::mutex>* glock = nullptr;
// Fargo Project: Revised by Xi ZHAO -- Nov 16, 2022

// For PVDLB 2023: FARGO: Fast Maximum Inner Product Search via Global Multi-Probing

// For any question, please feel free to contact me. Email: xzhaoca@connect.ust.hk

int main(int argc, char const* argv[])
{
	std::string dataset = "gist";
	if (argc > 1) {
		dataset = argv[1];
	}
	std::string argvStr[4];
	argvStr[1] = (dataset + ".data");
	argvStr[2] = (dataset + ".index");
	argvStr[3] = (dataset + ".ben");

	float c = 0.9f;
	int k = 50;
	int m, L, K;

	std::cout << "Using FARGO for " << argvStr[1] << std::endl;
	Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));
	std::vector<resOutput> res;
	m = 0;
	L = 5;
	K = 12;
	c = 0.8;

	Parameter param(prep, L, K, 1);

	lsh::timer timer;
	Partition parti(c, prep);
	// mf_alsh::Hash myslsh(prep, param, index_fold.append(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	myHNSW hnsw(prep, param, index_fold.append(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	hnsw.setEf(500);
	mariaV2 maria2(prep, param, index_fold.append(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	
	maria maria(prep, param, index_fold.append(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");

	mariaV3 maria3(prep, param, index_fold.append(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	
	int minsize_cl = 500;
	int num_cl = 10;
	int max_mst_degree = 3;
	//hcnngLite::hcnng<calInnerProductReverse>(dataset, prep.data, data_fold2 + argvStr[2] + "_hcnng", "index_result.txt", minsize_cl, num_cl, max_mst_degree, 0);

	hcnngLite::hcnng<calInnerProductReverse> hcnng(dataset, prep.data, data_fold2 + argvStr[2] + "_hcnng", "index_result.txt", 
		minsize_cl, num_cl, max_mst_degree, 1);

	res.push_back(Alg0_maria(hcnng, c, 100, k, L, K, prep));
	std::vector<int> ms = { 0,100,200,400,800,1200,1600,3200,6400};
	//ms = { 100 };
	res.push_back(Alg0_mariaV2(maria2, c, 100, k, L, K, prep));
	res.push_back(Alg0_maria(hnsw, c, 1000, k, L, K, prep));
	res.push_back(Alg0_maria(maria, c, 100, k, L, K, prep));
	res.push_back(Alg0_maria(maria3, c, 100, k, L, K, prep));
	// for (auto& x : ms) {
	// 	m = x + k;
	// 	res.push_back(Alg0_mfalsh(myslsh, c, m, k, L, K, prep));
	// 	//res.push_back(Alg0_maria(maria, c, m, k, L, K, prep));
	// 	//res.push_back(Alg0_HNSW(hnsw, c, m, k, L, K, prep));
	
	// }

	// for (auto& x : ms) {
	// 	m = x + k;
	// 	// res.push_back(Alg0_mfalsh(myslsh, c, m, k, L, K, prep));
	// 	//res.push_back(Alg0_maria(maria, c, m, k, L, K, prep));
	// 	res.push_back(Alg0_HNSW(hnsw, c, m, k, L, K, prep));
	
	// }

	saveAndShow(c, k, dataset, res);

	return 0;
}
