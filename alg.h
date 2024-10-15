#pragma once
#include <string>
#include "Preprocess.h"
//#include "mf_alsh.h"
#include "performance.h"
#include "basis.h"
#include "hcnngLite.h"
// #include "hnswlib.h"
#include "maria.h"
#include "mf_alsh.h"
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

extern std::atomic<size_t> _G_COST;

struct resOutput
{
	std::string algName;
	int L;
	int K;
	float c;
	float time;
	float recall;
	float ratio;
	float cost;
	float kRatio;
	float qps;
};

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

#if defined(unix) || defined(__unix__)
inline void localtime_s(tm* result, time_t* time) {
	if (localtime_r(time, result) == nullptr) {
		std::cerr << "Error converting time." << std::endl;
		std::memset(result, 0, sizeof(struct tm));
	}
}
#endif

template <typename mariaVx>
inline resOutput Alg0_maria(mariaVx& maria, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;

	Performance<queryN> perform;

	int t = 100;

	size_t cost1 = _G_COST;
	int nq = t * Qnum;

	std::vector<queryN> qs;
	for (int j = 0; j < nq; j++) {
		qs.emplace_back(j / t, c_, k_, prep, m_);
	}

	lsh::progress_display pd(nq);

	lsh::timer timer1;
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < nq; j++) {
		maria.knn(&(qs[j]));
		++pd;
	}
	float qt = (float)(timer1.elapsed() * 1000);


	for (int j = 0; j < nq; j++) {
		perform.update(qs[j], prep);
	}


	float mean_time = (float)perform.time_total / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k_) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);


	cost1 = _G_COST - cost1;

	resOutput res;
	res.algName = maria.alg_name;
	res.L = -1;
	res.K = m_;
	res.c = c_;
	res.qps = (float)nq / (qt / 1000);
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	//res.cost = ((float)cost1) / ((long long)perform.num * (long long)maria.N);
	res.cost = ((float)cost1) / ((long long)perform.num);
	res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}

inline resOutput Alg0_mariaV2(mariaV2& maria, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	//lsh::progress_display pd(Qnum);
	Performance<queryN> perform;
	lsh::timer timer1;
	int t = 1;
	size_t cost1 = _G_COST;
	lsh::progress_display pd(Qnum * t);
	for (int j = 0; j < Qnum * t; j++)
	{
		queryN query(j / t, c_, k_, prep, m_);
		maria.knn(&query);
		perform.update(query, prep);
		++pd;
	}

	float mean_time = (float)perform.time_total / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k_) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);

	cost1 = _G_COST - cost1;

	resOutput res;
	res.algName = "MariaV2";
	res.L = -1;
	res.K = m_;
	res.c = c_;
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)cost1) / ((long long)perform.num);
	res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}

inline resOutput Alg0_HNSW(myHNSW& hnsw, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");
	hnsw.setEf(m_);
	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	lsh::progress_display pd(Qnum);
	Performance<queryN> perform;
	lsh::timer timer1;
	for (int j = 0; j < Qnum; j++)
	{
		queryN query(j, c_, k_, prep, m_);
		hnsw.knn(&query);
		perform.update(query, prep);
		++pd;
	}

	float mean_time = (float)perform.time_total / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k_) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);

	resOutput res;
	res.algName = "HNSW";
	res.L = -1;
	res.K = m_;
	res.c = c_;
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)perform.cost) / ((long long)perform.num * (long long)hnsw.N);
	res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}

inline resOutput Alg0_mfalsh(mf_alsh::Hash& myslsh, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	//std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	int t = 100;
	int nq = Qnum * t;
	//lsh::progress_display pd(nq);
	Performance<mf_alsh::Query> perform;
	lsh::timer timer1;


	// for (int j = 0; j < nq; j++) {
	// 	qs.emplace_back(j / t, c_, k_, myslsh, prep, m_);
	// }

	lsh::progress_display pd(nq);
	std::vector<mf_alsh::Query*> qs(nq);
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < nq; j++) {
		//mf_alsh::Query query(j / t, c_, k_, myslsh, prep, m_);
		qs[j] = new mf_alsh::Query(j / t, c_, k_, myslsh, prep, m_);
		++pd;
	}
	float qt = timer1.elapsed();


	for (int j = 0; j < nq; j++) {
		perform.update(*(qs[j]), prep);
	}

	// #pragma omp parallel for schedule(dynamic)
	// 	for (int j = 0; j < nq; j++) {
	// 		mf_alsh::Query query(j / t, c_, k_, myslsh, prep, m_);
	// 		perform.update(query, prep);
	// 		++pd;
	// 	}

	//float qt = timer1.elapsed();
	float mean_time = (float)perform.time_total / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k_) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);

	resOutput res;
	res.algName = "FARGO";
	res.L = myslsh.L;
	res.K = myslsh.K;
	res.c = c_;
	res.qps = (float)nq / qt;
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)perform.cost) / ((long long)perform.num);
	res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}

void saveAndShow(float c, int k, std::string& dataset, std::vector<resOutput>& res);
