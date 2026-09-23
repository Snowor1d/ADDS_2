"""Verification and validation of the crowd model.

Two different questions live here, and the literature keeps them apart.

Verification asks whether the model does what it was written to do. Validation
asks whether what it does resembles what people do. The pedestrian evacuation
field has settled on a set of standard test cases for both, published as
RiMEA's guideline for microscopic evacuation analysis, IMO MSC.1/Circ.1238 for
ships, and ISO 20414 as a verification and validation protocol. Reporting a
pass or fail table against those tests is the currency of the field, and it is
what this package produces.

Nothing here is a unit test. These are measurements: they take minutes, they
produce numbers to be compared against published values, and a failure is a
finding rather than a broken build. `tests/` holds a fast smoke version so the
harness cannot rot unnoticed.
"""
