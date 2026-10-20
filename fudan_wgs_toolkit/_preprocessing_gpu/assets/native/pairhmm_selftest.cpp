#include "pairhmm_internal.hpp"
#ifdef PHMM_HAS_CUDA
#include "pairhmm_batch_regression.hpp"
#endif
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
struct OwnedRead {
    std::string bases;
    std::vector<uint8_t> q, ins, del, gap;
    explicit OwnedRead(std::string b) : bases(std::move(b)), q(bases.size(), 30), ins(bases.size(), 45), del(bases.size(), 45), gap(bases.size(), 10) {}
    phmm_read view() const { return {reinterpret_cast<const uint8_t *>(bases.data()), q.data(), ins.data(), del.data(), gap.data(), static_cast<uint32_t>(bases.size())}; }
};
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
bool near(double a, double b, double tolerance) {
    return (std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tolerance) || (std::isinf(a) && a == b);
}
std::string assessed(double number, bool valid) {
    if(!valid) return "null";
    std::ostringstream s; s<<std::setprecision(12)<<number; return s.str();
}
int main(int argc, char **argv) {
    try {
        const bool request_gpu = argc > 1 && std::string(argv[1]) == "--gpu";
        char error[1024]{};
        uint64_t dense_calls=0,dense_pairs=0,dense_kernels=0; double dense_scalar_error=0,dense_double_error=0;
        OwnedRead one("A"); auto v = one.view();
        const std::string a("A"); phmm_haplotype h{reinterpret_cast<const uint8_t *>(a.data()), 1};
        double analytical = 0;
        require(phmm_cpu_double_log10(&v, &h, &analytical, error, sizeof(error)) == 0, error);
        require(std::abs(analytical - std::log10(0.9 * 0.999)) < 1e-12, "One-base analytical probability failed");
        std::mt19937 rng(41719); const char alphabet[] = "ACGT";
        std::string reference(220, 'A'); for (char &b : reference) b = alphabet[rng()%4];
        std::vector<std::string> hs{reference, reference, reference, reference};
        hs[1][52] = 'N'; hs[2].insert(80, "GGA"); hs[3].erase(110, 4);
        std::vector<OwnedRead> owned;
        for (unsigned n=0; n<24; ++n) {
            unsigned length=1+rng()%151, offset=rng()%(reference.size()-length+1);
            owned.emplace_back(reference.substr(offset,length)); auto &r=owned.back();
            for (unsigned i=0; i<length; ++i) {
                r.q[i]=6+rng()%40; r.ins[i]=20+rng()%40; r.del[i]=20+rng()%40; r.gap[i]=8+rng()%20;
                if(rng()%97==0) r.bases[i]='N';
            }
            if(n%3==0) r.bases[length/2]=alphabet[rng()%4];
        }
        owned.emplace_back(std::string(150,'A'));
        std::vector<phmm_read> reads; for(const auto &r:owned) reads.push_back(r.view());
        std::vector<phmm_haplotype> haps;
        for(const auto &s:hs) haps.push_back({reinterpret_cast<const uint8_t *>(s.data()),static_cast<uint32_t>(s.size())});
        size_t pairs=reads.size()*haps.size(); std::vector<double> oracle(pairs),fp32(pairs),gpu(pairs);
        double float_error=0,gpu_float_error=0,gpu_double_error=0; uint64_t fallback=0;
        for(size_t r=0;r<reads.size();++r) for(size_t j=0;j<haps.size();++j) {
            size_t p=r*haps.size()+j;
            require(phmm_cpu_double_log10(&reads[r],&haps[j],&oracle[p],error,sizeof(error))==0,error);
            float scaled=phmm::scalar_scaled(reads[r],haps[j],phmm::float_tables(),std::ldexp(1.0f,120));
            if(scaled<1e-28f||!std::isfinite(scaled)){fp32[p]=oracle[p];++fallback;}
            else fp32[p]=static_cast<double>(::log10f(scaled)-::log10f(std::ldexp(1.0f,120)));
            require(near(fp32[p],oracle[p],8e-5),"Scalar FP32 disagrees with double oracle");
            if(std::isfinite(fp32[p]))float_error=std::max(float_error,std::abs(fp32[p]-oracle[p]));
        }
        OwnedRead masked=owned[5]; double before=phmm::cpu_double(masked.view(),haps[0]);
        for(auto *array:{&masked.q,&masked.ins,&masked.del,&masked.gap})for(auto &q:*array)q|=128;
        require(phmm::cpu_double(masked.view(),haps[0])==before,"Unsigned Phred &127 failed");
        phmm_stats counters{}; bool gpu_executed=false;
#ifdef PHMM_HAS_CUDA
        if(request_gpu) {
            phmm_cuda_context *c=nullptr;
            require(phmm_cuda_create(0,32u<<20,7,&c,error,sizeof(error))==0,error);
            std::unique_ptr<phmm_cuda_context,decltype(&phmm_cuda_destroy)> guard(c,phmm_cuda_destroy);
            require(phmm_cuda_compute(c,reads.data(),reads.size(),haps.data(),haps.size(),0,gpu.data(),&counters,error,sizeof(error))==0,error);
            require(counters.gpu_pairs==pairs&&counters.gpu_kernel_launches==(pairs+6)/7,"All-pair/remainder GPU counters failed");
            require(counters.cpu_double_fallback_pairs==fallback,"GPU low-probability fallback count differs from scalar contract");
            for(size_t p=0;p<pairs;++p) {
                require(near(gpu[p],fp32[p],2e-5),"GPU disagrees with scalar FP32");
                require(near(gpu[p],oracle[p],8e-5),"GPU disagrees with independent double oracle");
                if(std::isfinite(gpu[p])){
                    gpu_float_error=std::max(gpu_float_error,std::abs(gpu[p]-fp32[p]));
                    gpu_double_error=std::max(gpu_double_error,std::abs(gpu[p]-oracle[p]));
                }
            }
            phmm_stats rejected{};
            require(phmm_cuda_compute(c,reads.data(),reads.size(),haps.data(),haps.size(),1,gpu.data(),&rejected,error,sizeof(error))!=0,"Double request silently accepted");
            require(rejected.gpu_pairs==0,"Rejected double request launched GPU work");
            const auto dense=batch_initialization_regression();
            dense_calls=dense.calls; dense_pairs=dense.pairs; dense_kernels=dense.kernel_launches;
            dense_scalar_error=dense.max_scalar_error; dense_double_error=dense.max_double_error; gpu_executed=true;
        }
#else
        require(!request_gpu,"CPU-only build cannot satisfy --gpu; no silent fallback");
#endif
        std::cout<<std::setprecision(12)<<"{\"status\":\"passed\",\"synthetic_component_test\":true,\"scientific_benchmark\":false,\"pairs\":"<<pairs
            <<",\"gpu_executed\":"<<(gpu_executed?"true":"false")<<",\"max_fp32_double_log10_error\":"<<float_error
            <<",\"max_gpu_scalar_fp32_log10_error\":"<<assessed(gpu_float_error,gpu_executed)<<",\"max_gpu_double_log10_error\":"<<assessed(gpu_double_error,gpu_executed)
            <<",\"expected_scalar_fallback_pairs\":"<<fallback<<",\"gpu_pairs\":"<<counters.gpu_pairs
            <<",\"gpu_kernel_launches\":"<<counters.gpu_kernel_launches<<",\"cpu_double_fallback_pairs\":"<<counters.cpu_double_fallback_pairs
            <<",\"dense_regression_calls\":"<<dense_calls<<",\"dense_regression_pairs\":"<<dense_pairs<<",\"dense_regression_kernel_launches\":"<<dense_kernels
            <<",\"dense_regression_max_scalar_error\":"<<dense_scalar_error<<",\"dense_regression_max_double_error\":"<<dense_double_error
            <<",\"gpu_dp_cells\":"<<counters.gpu_dp_cells<<",\"gpu_kernel_milliseconds\":"<<counters.gpu_kernel_milliseconds<<"}\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<"PairHMM selftest failed: "<<e.what()<<'\n';return 1;}
}
