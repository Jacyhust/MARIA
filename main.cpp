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
//#include "mf_alsh.h"
//#include "basis.h"
#include "alg.h"
#include "maria.h"
#include "bf.h"
#include "normAdjustedDG.h"
extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;
std::unique_lock<std::mutex>* glock = nullptr;
// Fargo Project: Revised by Xi ZHAO -- Nov 16, 2022

// For PVDLB 2023: FARGO: Fast Maximum Inner Product Search via Global Multi-Probing

// For any question, please feel free to contact me. Email: xzhaoca@connect.ust.hk

int main(int argc, char const* argv[])
{
	std::string dataset = "mnist";
	if (argc > 1) {
		dataset = argv[1];
	}
	std::string argvStr[4];
	argvStr[1] = (dataset);
	argvStr[2] = (dataset + ".index");
	argvStr[3] = (dataset + ".bench_graph");

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

	int minsize_cl = 500;
	int num_cl = 10;
	int max_mst_degree = 3;

	Parameter param(prep, L, K, 1);

	lsh::timer timer;
	Partition parti(c, prep);
	//solidAnglePartition sap(prep, param, index_fold + (argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	myHNSW hnsw(prep, param, index_fold+(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	//hnsw.setEf(500);
	// mariaV2 maria2(prep, param, index_fold.append(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	//mariaV4 mariaV4(prep, param, index_fold + (argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	//mariaV5 mariaV5(prep, param, index_fold + (argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	//maria maria(prep, param, index_fold+(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");

	//maria_hcnng maria_hc(prep, param, index_fold+(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	//res.push_back(Alg0_maria(maria_hc, c, 100, k, L, K, prep));

	//myNADG nadg(prep.data, 24, 500, 1000);
	//myNAPG napg(prep.data, 24, 500, 1000);
	// mariaV3 maria3(prep, param, index_fold.append(argvStr[2]), parti, data_fold2 + "MyfunctionXTheta.data");
	//hcnngLite::hcnng<calInnerProductReverse>(dataset, prep.data, data_fold2 + argvStr[2] + "_hcnng", "index_result.txt", minsize_cl, num_cl, max_mst_degree, 0);
	//hcnngLite::hcnng<calInnerProductReverse> hcnng(dataset, prep.data, data_fold2 + argvStr[2] + "_hcnng", "index_result.txt", 
	//	minsize_cl, num_cl, max_mst_degree, 1);
	//hcnngLite::hcnng<calInnerProductReverse> hcnng(dataset, prep.data, data_fold2 + argvStr[2] + "_hcnng", "index_result.txt",
	//	minsize_cl, num_cl, max_mst_degree, 1);
	//hcnngLite::hcnng<calInnerProductReverse> hcnng(dataset, prep.data, data_fold2 + argvStr[2] + "_hcnng", "index_result.txt",
	//	minsize_cl, num_cl, max_mst_degree, 1);

	//res.push_back(Alg0_maria(hcnng, c, 100, k, L, K, prep));
	//res.push_back(Alg0_maria(maria, c, 100, k, L, K, prep));
	
	//res.push_back(Alg0_maria(nadg, c, 100, k, L, K, prep));
	//res.push_back(Alg0_maria(napg, c, 100, k, L, K, prep));
	//res.push_back(Alg0_maria(mariaV4, c, 100, k, L, K, prep));
	//res.push_back(Alg0_maria(mariaV5, c, 100, k, L, K, prep));
	res.push_back(Alg0_maria(hnsw, c, 100, k, L, K, prep));
	std::vector<int> ms = { 0,100,200,400,800,1200,1600,3200,6400};
	saveAndShow(c, k, dataset, res);

	return 0;
}
