/**
 * @file basis.h
 *
 * @brief A set of basic tools.
 */
#pragma once
#include <string>
#include <iostream>
#include <time.h>
#include <algorithm>
#include "fastL2_ip.h"
#include <chrono>

namespace lsh
{
	class progress_display
	{
		public:
		explicit progress_display(
			int long expected_count,
			std::ostream& os = std::cout,
			const std::string& s1 = "\n",
			const std::string& s2 = "",
			const std::string& s3 = "")
			: m_os(os), m_s1(s1), m_s2(s2), m_s3(s3)
		{
			restart(expected_count);
		}
		void restart(int long expected_count)
		{
			_count = _next_tic_count = _tic = 0;
			_expected_count = expected_count;
			m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
				<< m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
				<< std::endl
				<< m_s3;
			if (!_expected_count) {
				_expected_count = 1;
			}
		}
		int long operator += (int long increment) {
			if ((_count += increment) >= _next_tic_count)
			{
				display_tic();
			}
			return _count;
		}
		int long  operator ++ () {
			return operator += (1);
		}
		int long count() const {
			return _count;
		}
		int long expected_count() const {
			return _expected_count;
		}
		private:
		std::ostream& m_os;
		const std::string m_s1;
		const std::string m_s2;
		const std::string m_s3;
		int long _count, _expected_count, _next_tic_count;
		int _tic;
		void display_tic()
		{
			int tics_needed = int((double(_count) / _expected_count) * 50.0);
			do
			{
				m_os << '*' << std::flush;
			} while (++_tic < tics_needed);
			_next_tic_count = int((_tic / 50.0) * _expected_count);
			if (_count == _expected_count)
			{
				if (_tic < 51) m_os << '*';
				m_os << std::endl;
			}
		}
	};

	/**
	 * A timer object measures elapsed time, and it is very similar to boost::timer.
	 */
	class timer
	{
		public:
		timer() : time_begin(std::chrono::steady_clock::now()) {};
		~timer() {};
		/**
		 * Restart the timer.
		 */
		void restart()
		{
			time_begin = std::chrono::steady_clock::now();
		}
		/**
		 * Measures elapsed time.
		 *
		 * @return The elapsed time
		 */
		double elapsed()
		{
			std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
			return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) * 1e-6;// / CLOCKS_PER_SEC;
		}
		private:
		std::chrono::steady_clock::time_point time_begin;
	};
}

#include "distances_simd_avx512.h"
#include "patch_ubuntu.h"
extern std::atomic<size_t> _G_COST;

inline float cal_inner_product(float* v1, float* v2, int dim)
{
	++_G_COST;
#ifdef __AVX2__
	// printf("here!\n");
	// exit(-1);
	return faiss::fvec_inner_product_avx512(v1, v2, dim);
#else
	float res = 0.0;
	for (int i = 0; i < dim; ++i) {
		res += v1[i] * v2[i];
	}
	return res;

	return calIp_fast(v1, v2, dim);
#endif

}

inline float cal_cosine_similarity(float* v1, float* v2, int dim,
	float norm1, float norm2)
{
	++_G_COST;
#ifdef __AVX2__
	// printf("here!\n");
	// exit(-1);
	return faiss::fvec_inner_product_avx512(v1, v2, dim) / (norm1 * norm2);
#else
	float res = 0.0;
	for (int i = 0; i < dim; ++i) {
		res += v1[i] * v2[i];
	}
	return res / (norm1 * norm2);

	return calIp_fast(v1, v2, dim) / (norm1 * norm2);
#endif

}

inline float cal_L2sqr(float* v1, float* v2, int dim)
{
	++_G_COST;
#ifdef __AVX2__
	return (faiss::fvec_L2sqr_avx512(v1, v2, dim));
#else
	float res = 0.0;
	for (int i = 0; i < dim; ++i) {
		res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return res;
#endif

}

template <class T>
void clear_2d_array(T** array, int n)
{
	for (int i = 0; i < n; ++i) {
		delete[] array[i];
	}
	delete[] array;
}

inline float calInnerProductReverse(float* v1, float* v2, int dim) {
	return -cal_inner_product(v1, v2, dim);
}


// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <iomanip>
// #include <cstdio>
#include <unistd.h>
/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
inline size_t getCurrentRSS() {
#if defined(_WIN32)
	/* Windows -------------------------------------------------- */
	PROCESS_MEMORY_COUNTERS info;
	GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
	return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
	/* OSX ------------------------------------------------------ */
	struct mach_task_basic_info info;
	mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
		(task_info_t)&info, &infoCount) != KERN_SUCCESS)
		return (size_t)0L;      /* Can't access? */
	return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
	/* Linux ---------------------------------------------------- */
	long rss = 0L;
	FILE* fp = NULL;
	if ((fp = fopen("/proc/self/statm", "r")) == NULL)
		return (size_t)0L;      /* Can't open? */
	if (fscanf(fp, "%*s%ld", &rss) != 1) {
		fclose(fp);
		return (size_t)0L;      /* Can't read? */
	}
	fclose(fp);
	return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
	/* AIX, BSD, Solaris, and Unknown OS ------------------------ */
	return (size_t)0L;          /* Unsupported. */
#endif
}