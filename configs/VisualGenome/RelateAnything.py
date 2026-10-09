"""Official-pack supplied-region evaluation; not standard OpenSGG PredCls."""
method = "RelateAnything"
task = "supplied_regions"
protocol = "A1"
img_size = 448
eval_budget = 500
max_objects = 100
weights = "ema"
graph_constraint = True
split = "test"
limit = 0
metrics = [f"regions_A1_{metric}@{k}" for metric in ("R", "mR", "F1") for k in (20, 50, 100)]
