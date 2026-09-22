"""VG150 predicate-ontology cross-pair generalization diagnostic.

This subpackage answers one question: does the VG150 predicate ontology support
visual relation generalization across object pairs -- i.e. can a model that saw
``person-riding-horse`` and ``child-riding-bike`` predict ``riding`` for
``person-riding-bike`` from visual evidence rather than an object-pair prior?

It is a **dataset/ontology diagnostic, not a reproduction** of any published
method.  Nothing here loads a pretrained SGG checkpoint or reports a benchmark
number; every artifact carries :data:`common.STATUS_BLOCK`.
"""
