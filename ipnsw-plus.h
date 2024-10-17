#pragma once
#include "hnswlib.h"
#include <mutex>
#include <algorithm>
#include <fstream>
using hnsw = hnswlib::HierarchicalNSW<float>;

class ipNSW_plus {
    private:
    std::string index_ip, index_ang;
    IpSpace* ips = nullptr;
    hnsw* apg_ip = nullptr;
    hnsw* apg_ang = nullptr;
    //Preprocess* prep = nullptr;
    Data data;
    float* norms = nullptr;
    std::vector<int> hnsw_maps;//maps between hnsw internel labels and external labels
    float indexing_time = 0;
    public:
    int N;
    int dim;
    // // Number of hash functions
    // int S;
    // //#L Tables; 
    // int L;
    // // Dimension of the hash table
    // int K;
    //std::string index_file;
    std::string alg_name = "ipNSW-plus";

    ipNSW_plus(Preprocess& prep, Parameter& param_, const std::string& file) {
        reset(prep, param_, file, 1);
    }

    inline bool exists_test(const std::string& name) {
        //return false;
        std::ifstream f(name.c_str());
        return f.good();
    }

    void reset(Preprocess& prep, Parameter& param_, const std::string& file, bool isbuilt = true) {
        N = param_.N;
        dim = param_.dim;
        data = prep.data;
        index_ip = file;
        index_ang = file + "_plus_ang";
        norms = prep.norms;

        if (!(isbuilt && exists_test(index_ip))) {
            buildIndex(apg_ip, index_ip, false);
            std::cout << "Actual memory usage: " << getCurrentRSS() / (1024 * 1024) << " Mb \n";
            std::cout << "Build time:" << indexing_time << "  seconds.\n";
            FILE* fp = nullptr;
            fopen_s(&fp, "./indexes/ipnsw_plus_info.txt", "a");
            if (fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_ip.c_str(), (float)getCurrentRSS() / (1024 * 1024), indexing_time);
        }

        if (!(isbuilt && exists_test(index_ang))) {
            //delete apg_ip before building apg_ang for saving memory
            delete apg_ip;
            apg_ip = nullptr;

            buildIndex(apg_ang, index_ang, true);
            std::cout << "Actual memory usage: " << getCurrentRSS() / (1024 * 1024) << " Mb \n";
            std::cout << "Build time:" << indexing_time << "  seconds.\n";
            FILE* fp = nullptr;
            fopen_s(&fp, "./indexes/ipnsw_plus_info.txt", "a");
            if (fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_ang.c_str(), (float)getCurrentRSS() / (1024 * 1024), indexing_time);
        }
        else {
            //the indexes have been built now, delete data array for saving memory.
            //delete[] prep.data.base;

            std::cout << "Loading index from " << index_ang << ":\n";
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            ips = new IpSpace(dim);
            apg_ang = new hnsw(ips, index_ang, false);
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
        }

        if (!apg_ip) {
            std::cout << "Loading index from " << index_ip << ":\n";
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            ips = new IpSpace(dim);
            apg_ip = new hnsw(ips, index_ip, false);
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
        }
    }

    void setEf(size_t ef) {
        apg_ip->setEf(ef);
    }

    // int getM() {
    //     //return apg->maxM0_;
    // }

    void buildIndex(hnsw*& apg, const std::string& file, bool normalize) {
        int M = 24;
        int efC = 80;
        if (normalize) {
            M = 10;
            efC = 32;
        }
        ips = new IpSpace(dim);
        //apg = new hnsw[parti.numChunks];
        size_t report_every = N / 20;
        if (report_every > 1e5) report_every = N / 100;
        lsh::timer timer, timer_total;
        int j1 = 0;
        apg = new hnsw(ips, N, M, efC);
        auto id = 0;
        auto data0 = data.val[id];
        if (normalize) {
            data0 = new float[dim];
            for (int l = 0;l < dim;++l)
                data0[l] = data.val[id][l] / norms[id];
        }
        apg->addPoint((void*)(data0), (size_t)id);
        if (normalize) {
            delete data0;
        }
        std::mutex inlock;

        auto vecsize = N;

#pragma omp parallel for schedule(dynamic,256)
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
            float* data0 = data.val[j2];
            if (normalize) {
                data0 = new float[dim];
                for (int l = 0;l < dim;++l)
                    data0[l] = data.val[j2][l] / norms[j2];
            }
            apg->addPoint((void*)(data0), (size_t)j2);
            if (normalize) {
                delete data0;
            }
        }

        std::cout << "100%, Finish building HNSW\n";

        indexing_time = timer_total.elapsed();
        apg->saveIndex(file);
    }


    void knn(queryN* q) {
        lsh::timer timer;
        timer.restart();
        int ef = apg_ip->ef_;
        //ef = 200;

        std::vector<unsigned int> eps;
        //eps.push_back(0);
        int k_prime = 10;

        auto& appr_alg = apg_ang;
        auto res = appr_alg->searchKnn(q->queryPoint, k_prime + ef / 10);

        while (res.size() > k_prime) res.pop();
        //while (res.size() > 1) res.pop();

        //eps.re()
        while (!res.empty()) {
            auto top = res.top();
            eps.push_back(top.second);
            res.pop();
        }

        res = apg_ip->searchBaseLayerST<false>(eps, q->queryPoint, (size_t)(q->k) + ef);
        // auto res = apg_ip->searchBaseLayerST<false>(0, q->queryPoint, (size_t)(q->k) + ef);
        res = apg_ip->searchKnn(q->queryPoint, (size_t)(q->k) + ef);

        while (!res.empty()) {
            auto top = res.top();
            res.pop();
            q->resHeap.emplace(top.second, top.first);
            while (q->resHeap.size() > q->k) q->resHeap.pop();
        }

        while (!q->resHeap.empty()) {
            auto top = q->resHeap.top();
            q->resHeap.pop();
            q->res.emplace_back(top.id, 1.0f - top.dist);
        }
        std::reverse(q->res.begin(), q->res.end());
        q->time_total = timer.elapsed();
    }

    ipNSW_plus() {
        delete apg_ip;
        delete apg_ang;
        delete ips;
    }
};