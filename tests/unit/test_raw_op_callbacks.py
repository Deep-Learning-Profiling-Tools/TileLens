import pytest

from triton_viz.core.callbacks import OpCallbacks
from triton_viz.core.data import Store
from triton_viz.core.frontend.base import get_frontend
from triton_viz.core.patch import PatchOp


@pytest.mark.parametrize("adapted", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_raw_hook_preserves_operands_and_existing_callbacks(adapted, override):
    frontend = get_frontend("triton")
    seen = []
    ptr, value, mask = object(), object(), object()

    def forbidden_adapter(*args, **kwargs):
        raise AssertionError("Raw-only hooks must not invoke the adapter")

    wrapper = PatchOp(
        op=lambda *args, **kwargs: "original",
        op_type=Store,
        callbacks=OpCallbacks(
            before_callback=(lambda *args, **kwargs: seen.append(("before", args)))
            if adapted
            else None,
            after_callback=(lambda ret, *args, **kwargs: seen.append(("after", args)))
            if adapted
            else None,
            op_overrider=(lambda *args, **kwargs: "override") if override else None,
            raw_after_callback=lambda ret, *args, **kwargs: seen.append(
                ("raw", ret, args, kwargs)
            ),
        ),
        adapter=frontend.adapters[Store] if adapted else forbidden_adapter,
        run_op_overrider=frontend.run_op_overrider,
        maybe_yield_for_multism=lambda: None,
    )
    result = wrapper(ptr, value, mask=mask)
    assert result == ("override" if override else "original")
    expected = (
        [("before", (ptr, mask, None)), ("after", (ptr, mask, None))] if adapted else []
    )
    assert seen == expected + [("raw", result, (ptr, value), {"mask": mask})]
