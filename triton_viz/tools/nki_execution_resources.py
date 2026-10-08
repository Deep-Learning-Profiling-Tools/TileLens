"""Opt-in shared execution constraints from explicit memory provenance.

Resource groups are hypotheses to validate, not inferred hardware facts. A group
reserves its resource for the whole instruction; no service-time identification
or partial streaming occupancy is implied.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryExecutionResource:
    name: str
    engines: tuple[str, ...]
    memories: tuple[str, ...]

    def __post_init__(self):
        if (not self.name or len(set(self.engines)) < 2 or not self.memories
                or any(m not in {'sbuf', 'psum', 'hbm'} for m in self.memories)):
            raise ValueError('named multi-engine resource and explicit memories required')


def annotate_memory_resources(events, groups):
    """Annotate a copied event stream using only explicit transfer buffer types.

    Unknown or inconsistent memory provenance is an error for an affected
    engine. We do not decode pointer values, assume an unknown tile is SBUF,
    or derive a type from the engine whose contention is being tested.
    """
    storage_memory = {}
    for event in events:
        if event.get('op') != 'transfer':
            continue
        for prefix, field in [('src', 'mem_src'), ('dst', 'mem_dst')]:
            storage = event.get(prefix + '_storage')
            memory = event.get(field)
            if storage is None or memory not in {'sbuf', 'psum', 'hbm'}:
                continue
            if storage in storage_memory and storage_memory[storage] != memory:
                raise ValueError('conflicting storage memory provenance')
            storage_memory[storage] = memory
    names = [g.name for g in groups]
    if len(set(names)) != len(names):
        raise ValueError('unique resource names required')
    for event in events:
        resources = []
        for group in groups:
            if event.get('engine') not in group.engines:
                continue
            if event.get('op') == 'transfer':
                memories = [event.get('mem_src'), event.get('mem_dst')]
            else:
                storages = list(event.get('input_storages') or [])
                if event.get('output_storage') is not None:
                    storages.append(event['output_storage'])
                if not storages or any(s not in storage_memory for s in storages):
                    raise ValueError('explicit memory provenance missing for affected compute')
                memories = [storage_memory[s] for s in storages]
            if any(m not in {'sbuf', 'psum', 'hbm'} for m in memories):
                raise ValueError('explicit transfer memory provenance required')
            if set(memories).intersection(group.memories):
                resources.append(group.name)
        event['execution_resources'] = resources
    return dict(storage_memory=storage_memory, resource_names=names)
