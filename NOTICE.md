# Algorithm attribution and licensing

Fudan WGS Toolkit is distributed under GPL-3.0-only. Its Python implementations
of weighted rare-variant score tests, annotation masks, Cauchy combinations,
saddlepoint tails, and mixed-model arithmetic follow scientific algorithms by
Xihao Li, Zilin Li, Yuxin Yuan, and their collaborators.

Original algorithm sources and GPL attribution:

- [STAAR](https://github.com/li-lab-genetics/STAAR)
- [STAARpipeline](https://github.com/li-lab-genetics/STAARpipeline)
- [STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)
- [Frozen statistical reference](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05)

These names identify the scientific sources. They are not installed namespaces
or runtime dependencies of association analysis. The preprocessing entry instead
invokes established Conda-installed tools and includes an optional project
C++/CUDA/Java PairHMM adapter. Its algorithm sources and component licenses are
listed in [the preprocessing notices](fudan_wgs_toolkit/_preprocessing_gpu/THIRD_PARTY_NOTICES.md).
The package does not bundle external caller binaries, participant data,
downloaded annotations, or computed association tables. BWA-MEM2's optional
builder fetches fixed, verified upstream source and compiles it outside the
Conda environment; its upstream license remains with that downloaded source.

PLINK file formats are documented by the [PLINK project](https://www.cog-genomics.org/plink/1.9/formats).
The association converter reads binary files with Python and NumPy.
The single-subject preprocessing entry invokes Conda PLINK 2 and independently
checks every exported hard call, allele direction, sample ID and row order.
The independent FAVOR Essential annotation dataset is CC0-1.0, according to its
[Dataverse record](https://doi.org/10.7910/DVN/1VGTJI). Downloaded files remain external
inputs and must retain their own provenance and checksums.
