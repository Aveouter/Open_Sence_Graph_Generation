"""Phase I representation endpoints.

Three evaluators, in the order the plan ranks them. ``ccgp`` is the headline;
``role_equivariance`` and ``transplant`` are the other two primary endpoints. All
three are validated against synthetic latents whose answer is known before they
are pointed at a model, because an evaluator that has never been shown a case it
must fail cannot certify anything.
"""
