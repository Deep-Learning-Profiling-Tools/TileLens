"""cuTile shape, stride, and view behavior."""
# ruff: noqa: E402
import operator

import numpy as np
import pytest

ct = pytest.importorskip("cuda.tile")

from triton_viz.core.simulation.cutile import Array, Tile, cutile_builder


def test_tile_shape_operations():
    tile = cutile_builder.arange(16).reshape((4, -1))
    np.testing.assert_array_equal(tile.transpose().data, tile.data.T)
    np.testing.assert_array_equal(tile.permute((1, 0)).data, tile.data.T)
    np.testing.assert_array_equal(tile.extract((1, 0), (2, 2)).data, [[8, 9], [12, 13]])
    assert tile[:, None, ...].shape == (4, 1, 4)
    assert cutile_builder.expand_dims(tile, -1).shape == (4, 4, 1)
    assert cutile_builder.broadcast_to(cutile_builder.arange(4), (2, 4)).shape == (2, 4)
    assert tile.astype(ct.float32).dtype == ct.float32
    assert operator.index(cutile_builder.full((1,), 3, ct.int32).item()) == 3
    with pytest.raises(TypeError, match="extract"):
        tile[1:3]
    with pytest.raises(ValueError):
        cutile_builder.full((3,), 0, ct.float32)
    with pytest.raises(ValueError):
        tile.extract((2, 0), (2, 2))


@pytest.mark.parametrize(
    "view",
    [
        lambda a: a.T,
        lambda a: a[:, ::2],
        lambda a: a[::-1, ::-1],
        lambda a: np.broadcast_to(a[:1, :], (4, 8)),
    ],
)
def test_host_strides_and_order(view):
    data = view(np.arange(32, dtype=np.float32).reshape(4, 8))
    array = Array(data)
    assert array.strides == tuple(s // data.itemsize for s in data.strides)
    for order in ("C", "F", (1, 0)):
        expected = data if order == "C" else data.T
        tile = cutile_builder.load(array, (0, 0), (2, 4), order=order)
        np.testing.assert_array_equal(tile.data, expected[:2, :4])


def test_views_alias_storage_and_traversal():
    storage = np.arange(32, dtype=np.float32).reshape(4, 8)
    array = Array(storage)
    view = array.slice(0, 1, 4).slice(-1, Tile(2), Tile(7))
    assert view.shape == (3, 5)
    assert np.shares_memory(view.data, storage)
    tiled = view.tiled_view(
        (2, 4), padding_mode=ct.PaddingMode.ZERO, traversal_steps=(1, 2)
    )
    assert (tiled.num_tiles(0), tiled.num_tiles(1)) == (3, 3)
    np.testing.assert_array_equal(
        tiled.load((1, 1)).data, [[20, 21, 22, 0], [28, 29, 30, 0]]
    )
    tiled.store((2, 2), 99.0)
    assert storage[3, 6] == 99
    assert storage[3, 7] == 31
    with pytest.raises(ValueError):
        array.slice(0, -1, 2)
    with pytest.raises(ValueError):
        array.tiled_view((2,))
    with pytest.raises(ValueError):
        array.tiled_view((2, 4), traversal_steps=(0, 4))


def test_permuted_store_and_scalar_access():
    data = np.zeros((4, 8), np.float32)
    cutile_builder.store(
        Array(data), (1, 0), cutile_builder.full((4, 2), 7, ct.float32), order="F"
    )
    np.testing.assert_array_equal(data[:2, 4:], 7)
    assert not data[2:, :].any()
    cutile_builder.store(Array(data), (3, 7), Tile(11))
    assert cutile_builder.load(Array(data), (3, 7), ()).data == 11
    with pytest.raises(ValueError):
        cutile_builder.load(Array(data), (0,), (4,))


@pytest.mark.parametrize(
    "mode,expected",
    [
        (ct.PaddingMode.ZERO, 0),
        (ct.PaddingMode.NEG_ZERO, -0.0),
        (ct.PaddingMode.NAN, np.nan),
        (ct.PaddingMode.POS_INF, np.inf),
        (ct.PaddingMode.NEG_INF, -np.inf),
    ],
)
def test_padding_modes(mode, expected):
    value = cutile_builder.load(
        Array(np.arange(3, dtype=np.float32)), 0, 4, padding_mode=mode
    )
    np.testing.assert_equal(value.data[-1], expected)
    if mode == ct.PaddingMode.NEG_ZERO:
        assert np.signbit(value.data[-1])
