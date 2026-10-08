# Experimental shared execution resources

`CostModel.exclusive_execution_resources` is opt-in and empty by default. An
event's explicit `execution_resources` list reserves each declared resource from
start to issue completion, in addition to engine availability and RAW/WAR/WAW.
Per-engine work is unchanged. Resource wait is reported separately; it is a
scheduling-model quantity, not an identified native stall counter.

`annotate_memory_resources` binds hypotheses to typed source storage using
explicit transfer provenance. Unknown/conflicting types fail; numeric pointers
are never decoded as hardware-bank identities. The whole-instruction reservation
assumes a nonstreaming, exclusive resource and can overestimate conflicts.

For example, a **hypothesized** shared PSUM resource can make both relocation and
earlier extraction necessary for a Scalar branch to overlap an unrelated Vector
branch. With engine independence alone that companion appears unnecessary.
This synthetic test is a software-mechanism example, not evidence that actual
NeuronCore-v2 Vector/Scalar PSUM accesses serialize. Independent native controls
must distinguish that hypothesis from ordinary independent execution before
target prediction or agent experiments use it. Rejection leaves the default
model and old experimental gates unchanged.
