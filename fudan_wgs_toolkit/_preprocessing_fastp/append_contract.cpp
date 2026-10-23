// Generated formatter/ownership contracts, not a biological or speed benchmark.
// Compile against ORIGINAL with -DFASTP_APPEND_PATCHED=0 and PATCHED with =1.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef FASTP_READ_SOURCE
#error "Pass an explicitly bound read.cpp via FASTP_READ_SOURCE"
#endif
#ifndef FASTP_APPEND_PATCHED
#error "Pass 0 for original or 1 for patched"
#endif
static bool fail_allocation = false;
static bool count_allocation = false;
static size_t allocation_count = 0;
void* operator new(std::size_t n) {
    if (count_allocation) ++allocation_count;
    if (fail_allocation) throw std::bad_alloc();
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
#include FASTP_READ_SOURCE
static unsigned checks = 0;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
    ++checks;
}
static void append(Read& read, std::string& target) {
#if FASTP_APPEND_PATCHED
    read.appendToString(&target);
#else
    target += read.toString();
#endif
}
int main() {
    try {
        const size_t sizes[] = {0, 1, 15, 150, 1024, 100000};
        for (size_t n : sizes) {
            Read read("@read with spaces /1", std::string(n, 'A'), "+original plus", std::string(n, 'I'));
            const std::string prefix("before\0after", 12);
            std::string target = prefix;
            const std::string expected = prefix + read.toString();
            append(read, target);
            require(target == expected, "ordered FASTQ bytes differ");
            append(read, target);
            require(target == expected + read.toString(), "repeated append differs");
            require(read.mName == "@read with spaces /1" && read.mSeq.mStr == std::string(n, 'A')
                    && read.mStrand == "+original plus" && read.mQuality == std::string(n, 'I'), "read fields changed");
        }
        Read bytes(std::string("@n\0tag", 6), std::string("A\0CG", 4), std::string("+\rmeta", 6), std::string("I\0JJ", 4));
        std::string target("pre\0fix", 7);
        const std::string expected = target + bytes.toString();
        append(bytes, target);
        require(target == expected, "embedded NUL/CR/metadata bytes changed");
        Read long_name(std::string(8192, 'N'), "ACGT", std::string(8192, '+'), "IIII");
        target.clear(); append(long_name, target);
        require(target == long_name.toString(), "long metadata differs");
        Read normal("@contract/1", std::string(150, 'A'), "+kept", std::string(150, 'I'));
        target = "prefix"; target.reserve(4096);
        const std::string exact = target + normal.toString();
        allocation_count = 0; count_allocation = true;
        append(normal, target);
        count_allocation = false;
        require(target == exact, "reserved append content differs");
#if FASTP_APPEND_PATCHED
        require(allocation_count == 0, "reserved direct append allocated a temporary");
        require(readAppendLength(0, 0, 0, 0, 4) == 4, "empty exact capacity rejected");
        require(readAppendLength(1, 2, 3, 4, 14) == 14, "exact capacity rejected");
        bool denied = false;
        try { readAppendLength(0, 0, 0, 0, 3); } catch (const std::length_error&) { denied = true; }
        require(denied, "four separators overflow not rejected");
        denied = false;
        try { readAppendLength(1, 2, 3, 4, 13); } catch (const std::length_error&) { denied = true; }
        require(denied, "one byte shortage not rejected");
        denied = false;
        try { readAppendLength(std::numeric_limits<size_t>::max(), 0, 0, 0,
                               std::numeric_limits<size_t>::max()); }
        catch (const std::length_error&) { denied = true; }
        require(denied, "size_t overflow not rejected before allocation");
        require(readAppendLength(std::numeric_limits<size_t>::max()-4, 0, 0, 0,
                                 std::numeric_limits<size_t>::max()) == std::numeric_limits<size_t>::max(),
                "maximum arithmetic boundary differs");
        const std::string name = normal.mName, sequence = normal.mSeq.mStr,
                          plus = normal.mStrand, qualities = normal.mQuality;
        std::string* aliases[] = {&normal.mName, &normal.mSeq.mStr, &normal.mStrand, &normal.mQuality};
        for (std::string* alias : aliases) {
            denied = false;
            try { normal.appendToString(alias); } catch (const std::invalid_argument&) { denied = true; }
            require(denied && normal.mName == name && normal.mSeq.mStr == sequence
                    && normal.mStrand == plus && normal.mQuality == qualities, "invalid alias changed a source");
        }
        denied = false;
        try { normal.appendToString(NULL); } catch (const std::invalid_argument&) { denied = true; }
        require(denied, "null target not rejected");
        // Force the next string growth to fail; no target bytes may be exposed.
        std::string short_target = "unchanged";
        const std::string prior = short_target;
        fail_allocation = true;
        bool allocation_failed = false;
        try { normal.appendToString(&short_target); }
        catch (const std::bad_alloc&) { allocation_failed = true; }
        fail_allocation = false;
        require(allocation_failed && short_target == prior, "allocation failure partially changed the target");
        normal.appendToString(&short_target);
        require(short_target == prior + normal.toString(), "valid retry after failure differs");
#else
        require(allocation_count > 0, "baseline temporary-allocation contract not observed");
#endif
        std::cout << "PASS " << checks << " formatter contracts; source="
                  << (FASTP_APPEND_PATCHED ? "patched" : "original")
                  << "; generated inputs only; no throughput claim\n";
        return 0;
    } catch (const std::exception& error) {
        fail_allocation = false; count_allocation = false;
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
