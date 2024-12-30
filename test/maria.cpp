#include <iostream>
#include <memory>
#include <chrono>

//#include "../includes/indexDescdent.hpp"
//#include "../includes/RNNDescent.h"
//#include "../includes/utils/io.hpp"
#include "../includes/utils/StructType.h"
#include "../includes/utils/performance.h"

#include "../includes/maria_Pruning.h"

#if defined(unix) || defined(__unix__)

std::string data_fold = "./dataset/", index_fold = "./indexes/";
std::string data_fold1 = data_fold, data_fold2 = data_fold + ("MIPS/");
#else
std::string data_fold = "./dataset/", index_fold = " ";
std::string data_fold1 = data_fold;
std::string data_fold2 = data_fold + ("MIPS/");
#endif

int main(int argc, char* argv[])
{
    std::string dataset = "toy";
    int varied_n = 0;
    if (argc > 1) dataset = argv[1];
    if (argc > 2) varied_n = std::atoi(argv[2]);

    std::string argvStr[4];
    argvStr[3] = (dataset + ".bench_graph");
    std::cout << "Using MARIA for " << argvStr[1] << std::endl;


    float c = 0.9f;
    int L = 4;
    int K = 16;

    Preprocess prep(data_fold1 + dataset, data_fold2 + (argvStr[3]), varied_n);
    Partition parti(c, prep);

    if (varied_n > 0) dataset += std::to_string(varied_n);
    float c_ = 0.5;
    int k_ = 50;
    int M = 48;
    int recall = 0;
    float ratio = 0.0f;
    lsh::timer timer;
    auto times1 = timer.elapsed();
    lsh::timer timer11;
    int cost = 0;

    auto& queries = prep.queries;

    queries.N = 100;
    int repeat = 160;
#if defined(_DEBUG) || defined(_MSC_VER)
    repeat = 1;
#endif // _DEBUG
    int nq = queries.N * repeat;

    std::vector<queryN> qs;
    qs.reserve(nq);
    std::cout << "nq= " << nq << std::endl;
    for (int i = 0; i < nq; ++i) {
        qs.emplace_back(i % (queries.N), c_, k_, queries[i % (queries.N)], queries.dim, 1.0f);
    }

    std::vector<resOutput> res;
    std::vector<int> efs = { 0,10,20,30,40,50,75,100,150,200,250,300 };

    maria_noPruning maria(prep.data, prep.SquareLen, index_fold + dataset, parti, L, K);
    maria.buildFn();
    for (auto& ef : efs) {
        maria.ef = ef;
        res.push_back(searchFunctionFn(maria, qs, prep, 3));
    }

    for (auto& ef : efs) {
        maria.ef = ef;
        res.push_back(searchFunctionFn(maria, qs, prep, 4));
    }

    saveAndShow(c, k_, dataset, res);

    return 0;
}
