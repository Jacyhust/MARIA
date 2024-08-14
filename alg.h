#pragma once
#include <string>
#include "Preprocess.h"
#include "mf_alsh.h"
#include "performance.h"
#include "basis.h"
#include "hcnngLite.h"
// #include "hnswlib.h"
#include "maria.h"
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
};

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

#if defined(unix) || defined(__unix__)
inline void localtime_s(tm* ltm, time_t* now) {}
#endif

inline resOutput Alg0_mfalsh(mf_alsh::Hash& myslsh, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	lsh::progress_display pd(Qnum);
	Performance<mf_alsh::Query> perform;
	lsh::timer timer1;
	for (int j = 0; j < Qnum; j++)
	{
		mf_alsh::Query query(j, c_, k_, myslsh, prep, m_);
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
	res.algName = "FARGO";
	res.L = myslsh.L;
	res.K = myslsh.K;
	res.c=c_;
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)perform.cost) / ((long long)perform.num * (long long)myslsh.N);
	res.kRatio=perform.kRatio/perform.num;
	//delete[] ltm;
	return res;
}

template <typename mariaVx>
inline resOutput Alg0_maria(mariaVx& maria, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	
	Performance<queryN> perform;
	lsh::timer timer1;
	int t = 1;

	size_t cost1 = _G_COST;

	lsh::progress_display pd(Qnum*t);
	for (int j = 0; j < Qnum*t; j++)
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
	res.algName = maria.alg_name;
	res.L = -1;
	res.K = m_;
	res.c = c_;
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
	int t=1;
	size_t cost1 = _G_COST;
	lsh::progress_display pd(Qnum*t);
	for (int j = 0; j < Qnum*t; j++)
	{
		queryN query(j/t, c_, k_, prep, m_);
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

void saveAndShow(float c, int k, std::string& dataset, std::vector<resOutput>& res);
