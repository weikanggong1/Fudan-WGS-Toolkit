/* Generated boundary contracts only; not a biological benchmark or speed test.
 * Include the candidate file under distinct public names, then compare its
 * actual static cached-rescue function against the original upstream library.
 */
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define mem_pestat candidate_mem_pestat
#define mem_matesw candidate_mem_matesw
#define mem_pair candidate_mem_pair
#define mem_sam_pe candidate_mem_sam_pe
#ifndef BWA_PAIR_SOURCE
#error Explicit candidate source path required
#endif
#include BWA_PAIR_SOURCE
#undef mem_pestat
#undef mem_matesw
#undef mem_pair
#undef mem_sam_pe
extern int mem_matesw(const mem_opt_t*, const bntseq_t*, const uint8_t*, const mem_pestat_t*, const mem_alnreg_t*, int, const uint8_t*, mem_alnreg_v*);
extern mem_opt_t *mem_opt_init(void);
extern int bwa_verbose;

static uint32_t next_value(uint32_t *s) { *s=*s*1664525u+1013904223u;return *s; }
static uint8_t base(const uint8_t *pac, int p) { return (pac[p>>2]>>((~p&3)<<1))&3; }
static void same_regions(const mem_alnreg_v *a, const mem_alnreg_v *b) {
    size_t i;assert(a->n==b->n);
    for(i=0;i<a->n;i++)assert(memcmp(&a->a[i],&b->a[i],sizeof(mem_alnreg_t))==0);
}
static void *run_thread(void *argument) {
    uintptr_t tid=(uintptr_t)argument;
    const int lengths[]={1,18,19,31,75,151,300};
    int which, policy, anchor, reset_each;uint32_t rng=1837u+(uint32_t)tid;
    bntseq_t bns;uint8_t pac[1025];mem_opt_t *opt=mem_opt_init();
    memset(&bns,0,sizeof(bns));bns.l_pac=4096;bns.n_seqs=1;
    bns.anns=calloc(1,sizeof(bntann1_t));assert(bns.anns);
    bns.anns[0].offset=0;bns.anns[0].len=4096;bns.anns[0].name="contract_reference";
    memset(pac,0,sizeof(pac));
    for(which=0;which<4096;which++)pac[which>>2]|=(next_value(&rng)>>30)<<((~which&3)<<1);
    for(which=0;which<(int)(sizeof(lengths)/sizeof(*lengths));which++) {
        int length=lengths[which];uint8_t *query=malloc(length),*original_query=malloc(length),*cached_query=malloc(length);
        for(policy=0;policy<5;policy++)for(reset_each=0;reset_each<2;reset_each++) {
            mem_matesw_workspace_t workspace={{0,0},0};mem_alnreg_v old={0,0,0},candidate={0,0,0},api={0,0,0};
            mem_pestat_t pes[4];int dir;
            for(dir=0;dir<4;dir++){memset(&pes[dir],0,sizeof(pes[dir]));pes[dir].low=10;pes[dir].high=360;pes[dir].failed=policy==1||((policy==2)&&dir!=1);}
            for(anchor=0;anchor<12;anchor++) {
                mem_alnreg_t a;int n_old,n_new,n_api,i;
                if(anchor==0)for(i=0;i<length;i++) {
                    uint8_t b=base(pac,1100+i);
                    query[i]=policy==3?4:b;
                    if(policy==4)query[i]=b<4?3-base(pac,1100+length-1-i):4;
                }
                memcpy(original_query,query,length);memcpy(cached_query,query,length);
                memset(&a,0,sizeof(a));a.rid=0;a.rb=1000+anchor*7;a.re=a.rb+length;a.qe=length;a.score=length;
                if(reset_each){old.n=0;candidate.n=0;api.n=0;}
                n_old=mem_matesw(opt,&bns,pac,pes,&a,length,original_query,&old);
                n_new=mem_matesw_cached(opt,&bns,pac,pes,&a,length,cached_query,&candidate,&workspace);
                n_api=candidate_mem_matesw(opt,&bns,pac,pes,&a,length,cached_query,&api);
                assert(n_old==n_new&&n_old==n_api);same_regions(&old,&candidate);same_regions(&old,&api);
                assert(memcmp(original_query,query,length)==0&&memcmp(cached_query,query,length)==0);
            }
            if(policy==1)assert(workspace.profile[0]==0&&workspace.profile[1]==0&&workspace.reverse_query==0);
            mem_matesw_workspace_destroy(&workspace);
            assert(!workspace.profile[0]&&!workspace.profile[1]&&!workspace.reverse_query);
            mem_matesw_workspace_destroy(&workspace); /* NULL-safe cleanup. */
            free(old.a);free(candidate.a);free(api.a);
        }
        free(query);free(original_query);free(cached_query);
    }
    free(bns.anns);free(opt);return 0;
}
int main(void) {
    pthread_t workers[4];int i;bwa_verbose=0;
    for(i=0;i<4;i++)assert(pthread_create(&workers[i],0,run_thread,(void*)(uintptr_t)i)==0);
    for(i=0;i<4;i++)assert(pthread_join(workers[i],0)==0);
    puts("actual_cached_rescue_and_preserved_API_vs_upstream: 3360_generated_anchor_contracts; four threads; all exact; query restored; owners released");
    return 0;
}
