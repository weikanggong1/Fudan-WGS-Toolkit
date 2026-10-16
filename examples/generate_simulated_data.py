"""Generate direct prepared inputs; choose --preset small for an artificial test.

Reference defaults require a private aggregate calibration profile.
See docs/generate_simulated_data.md for parameters and analysis calls.
"""
from fudan_wgs_toolkit.simulate import main


if __name__ == "__main__":
    main()
