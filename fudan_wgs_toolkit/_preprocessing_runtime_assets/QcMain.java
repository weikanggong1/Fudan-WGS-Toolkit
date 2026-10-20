package org.fudan.wgs.preprocessing;

/** Run the original GATK CLI and exit after its normal, successful return. */
public final class QcMain extends org.broadinstitute.hellbender.Main {
    public static void main(final String[] args) {
        // mainEntry retains GATK argument handling, result handling, and all
        // exception exit codes. Its tools have completed their normal cleanup
        // before returning. A successful CLI need not retain idle R-output
        // reader threads from HTSJDK's non-daemon cached thread pool.
        new QcMain().mainEntry(args);
        System.exit(0);
    }
}
