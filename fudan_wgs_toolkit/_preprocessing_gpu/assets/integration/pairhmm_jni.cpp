#include <jni.h>
#include <cuda_runtime_api.h>
#include "pairhmm_cuda.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Context {
    phmm_cuda_context* backend = nullptr;
    uint64_t max_device_bytes = 0;
    uint32_t max_batch_pairs = 0;
    int max_native_threads = 0;
};
void fail(JNIEnv* env, const std::string& message) {
    if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message.c_str());
}
uint64_t setting(const char* name, uint64_t fallback, uint64_t maximum) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    std::size_t used = 0;
    unsigned long long value = std::stoull(raw, &used);
    if (used != std::string(raw).size() || value == 0 || value > maximum) throw std::runtime_error(std::string("Invalid bounded setting: ") + name);
    return value;
}
Context* context(jlong pointer) {
    if (!pointer) throw std::runtime_error("Missing CUDA PairHMM context");
    return reinterpret_cast<Context*>(static_cast<uintptr_t>(pointer));
}
std::string quoted(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c >= 32 && c < 127) out << c;
        else out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    }
    out << '"'; return out.str();
}
std::vector<uint8_t> bytes(JNIEnv* env, jobject holder, jfieldID field, size_t& allocated) {
    auto array = static_cast<jbyteArray>(env->GetObjectField(holder, field));
    if (!array || env->ExceptionCheck()) throw std::runtime_error("Missing PairHMM byte array");
    const jsize length = env->GetArrayLength(array);
    if (length <= 0 || allocated + static_cast<size_t>(length) > 128ULL * 1024 * 1024) {
        env->DeleteLocalRef(array); throw std::runtime_error("PairHMM JNI host input bound exceeded");
    }
    std::vector<uint8_t> result(static_cast<size_t>(length));
    env->GetByteArrayRegion(array, 0, length, reinterpret_cast<jbyte*>(result.data()));
    env->DeleteLocalRef(array);
    if (env->ExceptionCheck()) throw std::runtime_error("Could not copy PairHMM input");
    allocated += result.size(); return result;
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_intel_gkl_pairhmm_IntelPairHmm_createNative(JNIEnv* env, jclass, jint device, jint threads) {
    try {
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        if (!visible || !*visible || std::string(visible).find(',') != std::string::npos) throw std::runtime_error("CUDA_VISIBLE_DEVICES must explicitly select one authorized GPU");
        if (device != 0) throw std::runtime_error("Single-visible-GPU adapter requires local device ordinal 0");
        std::unique_ptr<Context> owner(new Context);
        owner->max_device_bytes = setting("FNIT_PAIRHMM_MAX_DEVICE_BYTES", 2ULL << 30, 18ULL << 30);
        owner->max_batch_pairs = static_cast<uint32_t>(setting("FNIT_PAIRHMM_MAX_BATCH_PAIRS",256,65536));
        owner->max_native_threads = threads;
        char error[2048] = {};
        const int code = phmm_cuda_create(device, owner->max_device_bytes, owner->max_batch_pairs, &owner->backend, error, sizeof(error));
        if (code || !owner->backend) throw std::runtime_error(std::string("CUDA PairHMM create failed: ") + error);
        return static_cast<jlong>(reinterpret_cast<uintptr_t>(owner.release()));
    } catch (const std::exception& error) { fail(env,error.what()); return 0; }
}

extern "C" JNIEXPORT void JNICALL
Java_com_intel_gkl_pairhmm_IntelPairHmm_computeNative(JNIEnv* env, jclass, jlong pointer,
        jobjectArray java_reads, jobjectArray java_haps, jdoubleArray output) {
    try {
        Context* owner = context(pointer);
        const jsize nr = env->GetArrayLength(java_reads), nh = env->GetArrayLength(java_haps);
        const uint64_t pairs = static_cast<uint64_t>(nr) * static_cast<uint64_t>(nh);
        if (nr <= 0 || nh <= 0 || pairs > 16ULL * 1024 * 1024 || pairs != static_cast<uint64_t>(env->GetArrayLength(output))) throw std::runtime_error("Invalid bounded PairHMM output shape");
        jclass read_class = env->FindClass("org/broadinstitute/gatk/nativebindings/pairhmm/ReadDataHolder");
        jclass hap_class = env->FindClass("org/broadinstitute/gatk/nativebindings/pairhmm/HaplotypeDataHolder");
        const char* names[] = {"readBases","readQuals","insertionGOP","deletionGOP","overallGCP"};
        std::array<jfieldID,5> fields{};
        for (size_t k=0;k<5;++k) fields[k]=env->GetFieldID(read_class,names[k],"[B");
        jfieldID hap_field=env->GetFieldID(hap_class,"haplotypeBases","[B");
        if (env->ExceptionCheck()) throw std::runtime_error("GATK native-holder field ABI mismatch");
        std::vector<std::array<std::vector<uint8_t>,5>> storage(static_cast<size_t>(nr));
        std::vector<std::vector<uint8_t>> hap_storage(static_cast<size_t>(nh));
        std::vector<phmm_read> reads(static_cast<size_t>(nr)); std::vector<phmm_haplotype> haps(static_cast<size_t>(nh));
        size_t allocated=0;
        for (jsize i=0;i<nr;++i) {
            jobject holder=env->GetObjectArrayElement(java_reads,i);
            if (!holder) throw std::runtime_error("Null PairHMM read holder");
            for(size_t k=0;k<5;++k) storage[i][k]=bytes(env,holder,fields[k],allocated);
            env->DeleteLocalRef(holder);
            size_t n=storage[i][0].size();
            for(size_t k=1;k<5;++k) if(storage[i][k].size()!=n) throw std::runtime_error("Inconsistent PairHMM read array lengths");
            reads[i]={storage[i][0].data(),storage[i][1].data(),storage[i][2].data(),storage[i][3].data(),storage[i][4].data(),static_cast<uint32_t>(n)};
        }
        for(jsize i=0;i<nh;++i) {
            jobject holder=env->GetObjectArrayElement(java_haps,i);
            if(!holder) throw std::runtime_error("Null PairHMM haplotype holder");
            hap_storage[i]=bytes(env,holder,hap_field,allocated); env->DeleteLocalRef(holder);
            haps[i]={hap_storage[i].data(),static_cast<uint32_t>(hap_storage[i].size())};
        }
        env->DeleteLocalRef(read_class);env->DeleteLocalRef(hap_class);
        std::vector<double> results(static_cast<size_t>(pairs));phmm_stats call_stats{};char error[2048]={};
        if(phmm_cuda_compute(owner->backend,reads.data(),static_cast<uint32_t>(nr),haps.data(),static_cast<uint32_t>(nh),0,results.data(),&call_stats,error,sizeof(error))) throw std::runtime_error(std::string("CUDA PairHMM compute failed: ")+error);
        env->SetDoubleArrayRegion(output,0,static_cast<jsize>(pairs),results.data());
    } catch(const std::exception& error) { fail(env,error.what()); }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_intel_gkl_pairhmm_IntelPairHmm_statsNative(JNIEnv* env,jclass,jlong pointer) {
    try {
        Context* owner=context(pointer);phmm_stats stats{};
        if(phmm_cuda_get_stats(owner->backend,&stats)) throw std::runtime_error("Cannot read CUDA provider counters");
        cudaDeviceProp properties{};
        cudaError_t code=cudaGetDeviceProperties(&properties,stats.device_ordinal);
        if(code!=cudaSuccess) throw std::runtime_error(std::string("CUDA device evidence failed: ")+cudaGetErrorString(code));
        std::ostringstream uuid;
        for(unsigned char value:properties.uuid.bytes) uuid<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(value);
        std::ostringstream out;out<<std::setprecision(17)<<"{\"provider\":"<<quoted(phmm_cuda_provider())
            <<",\"device_ordinal\":"<<stats.device_ordinal<<",\"device_name\":"<<quoted(properties.name)<<",\"device_uuid_hex\":"<<quoted(uuid.str())
            <<",\"compute_calls\":"<<stats.compute_calls<<",\"input_pairs\":"<<stats.input_pairs<<",\"gpu_pairs\":"<<stats.gpu_pairs
            <<",\"gpu_kernel_launches\":"<<stats.gpu_kernel_launches<<",\"cpu_double_fallback_pairs\":"<<stats.cpu_double_fallback_pairs
            <<",\"gpu_dp_cells\":"<<stats.gpu_dp_cells<<",\"h2d_bytes\":"<<stats.h2d_bytes<<",\"d2h_bytes\":"<<stats.d2h_bytes
            <<",\"gpu_kernel_milliseconds\":"<<stats.gpu_kernel_milliseconds<<",\"host_wall_milliseconds\":"<<stats.host_wall_milliseconds
            <<",\"owned_allocated_bytes\":"<<stats.owned_allocated_bytes<<",\"owned_peak_allocated_bytes\":"<<stats.owned_peak_allocated_bytes
            <<",\"invalid_gpu_result_pairs\":"<<stats.invalid_gpu_result_pairs
            <<",\"device_free_bytes_before\":"<<(stats.memory_snapshot_before_valid?std::to_string(stats.device_free_bytes_before):"null")
            <<",\"device_free_bytes_after\":"<<(stats.memory_snapshot_after_valid?std::to_string(stats.device_free_bytes_after):"null")
            <<",\"device_total_bytes\":"<<((stats.memory_snapshot_before_valid||stats.memory_snapshot_after_valid)?std::to_string(stats.device_total_bytes):"null")
            <<",\"max_device_bytes\":"<<owner->max_device_bytes<<",\"max_batch_pairs\":"<<owner->max_batch_pairs
            <<",\"requested_native_threads\":"<<owner->max_native_threads<<",\"precision\":\"FP32_with_counted_GKL_double_underflow_fallback\"}";
        return env->NewStringUTF(out.str().c_str());
    }catch(const std::exception& error){fail(env,error.what());return nullptr;}
}
extern "C" JNIEXPORT void JNICALL
Java_com_intel_gkl_pairhmm_IntelPairHmm_destroyNative(JNIEnv* env,jclass,jlong pointer) {
    try {std::unique_ptr<Context> owner(context(pointer));phmm_cuda_destroy(owner->backend);owner->backend=nullptr;}
    catch(const std::exception& error){fail(env,error.what());}
}
