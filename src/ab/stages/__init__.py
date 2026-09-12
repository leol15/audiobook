"""Pipeline stages, numbered in execution order. Each exposes run(paths, cfg, force) -> artifact
and, where cached, inputs(paths, cfg) -> dict of named input hashes for `ab status`."""
