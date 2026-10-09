# Downstream Tensor readiness transport

With experimental Tensor initiation enabled, timeline `end` is issue end and
`ready_end` is result availability. A downstream DAG scheduler must preserve
both values. Ordinary consumers wait for readiness; proven exact-range Tensor
accumulations may forward at issue end, retaining any intervening reader hazard.
Engine busy time remains initiation work, not the sum of readiness tails.

The simulation components now export `tensor_pipeline_startup_ns`, zero when
the experimental pipeline is disabled. It represents the same existing startup
used internally, allowing a downstream scheduler to retain initial Tensor
availability rather than assuming zero. This adds metadata; it does not change
the simulator's timing calculation or validate any native parameter.

The companion agent checks compare the actual CostModel and DAG scheduler for
startup, forwarding, conservative overwrite/read hazards and independent issue
overlap. Core module checks intentionally omit optional package UI/tracer
bootstrap; tracing and hardware transport require separate validation.
