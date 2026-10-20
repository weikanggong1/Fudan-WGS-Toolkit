# Optional preprocessing GPU sources and external dependencies

The project-written Java adapter, JNI glue, CUDA recurrence and build controller
are covered by the parent package's GPL-3.0-only license. They provide a separate
classpath adapter; they do not modify the user's original GATK jar.

External scientific interfaces and licenses, checked at their fixed versions:

- GATK 4.2.0.0, Copyright 2021 Broad Institute, Inc., Apache-2.0:
  https://github.com/broadinstitute/gatk/blob/4.2.0.0/LICENSE.TXT
- GKL 0.8.6, Copyright (c) 2016 Intel Corporation, MIT, with the upstream
  GATK PairHMM portion covered by BSD-3-Clause as identified in its README:
  https://github.com/IntelLabs/GKL/blob/0.8.6/LICENSE
  https://github.com/IntelLabs/GKL/blob/0.8.6/README.md
- gatk-native-bindings 1.0.0, Copyright (c) 2009-2016 GATK Authors,
  BSD-3-Clause:
  https://github.com/broadinstitute/gatk-native-bindings/blob/1.0.0/LICENSE.TXT

SOURCES.md records the exact algorithm/interface source revisions. No original
GATK/GKL source, jar/native binary, official sequence fixture, CUDA runtime,
reference genome, known sites, or participant sequence is included in this
source bundle. Install external dependencies through Conda and retain their
licenses. The builder links the environment's dynamic CUDA runtime; it does not
redistribute CUDA SDK/runtime files or change their licensing terms.

The original GKL 0.8.6 public test fixture is optional, remains external, and is
accepted only at SHA-256
598c5a3cedb9aaa10a0da258a9d6272ef51e4b59262dea90293fe0c6d9a86244.
The project does not bundle the independently mentioned gpuPairHMM project or
use its implementation in this backend.
