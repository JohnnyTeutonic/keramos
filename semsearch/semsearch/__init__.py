"""semsearch: architecture search over keramos documents by their graph, not their config.

canon        canonical form and hash (rule 38 plus order, dead code, spec-licensed aliases; class level)
fingerprint  behavioural fingerprints in keramos's reference interpreter
semdiff      typed edit script and distance between two documents
mutate       typed graph edits and crossover that always give valid documents
config_space a paperkiln-style config space and its lowering, for comparison
torch_exec   a trainable interpreter of keramos documents (CPU)
task         the proxy task (associative recall) and its training run
search       regularised evolution over the four representations
"""
