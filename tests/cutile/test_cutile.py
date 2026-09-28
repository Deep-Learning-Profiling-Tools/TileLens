"""Basic tests for the NumPy-backed cuTile interpreter."""

import numpy as np
import pytest

ct = pytest.importorskip("cuda.tile")

from triton_viz.core.simulation.cutile import (  # noqa: E402
    Array,
    CuTileInterpretedFunction,
    Tile,
    cutile_builder,
)


def test_tile_creation():
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    tile = Tile(data)

    assert tile.shape == (2, 2)
    assert tile.dtype == ct.int32
    np.testing.assert_array_equal(tile.data, data)

    # Test data ownership and immutability.
    data[0, 0] = 99
    assert tile.data[0, 0] == 1
    with pytest.raises(ValueError):
        tile.data[0, 0] = 99

    assert Tile(3).dtype == ct.int32
    assert Tile(3.0).dtype == ct.float32


def test_arithmetic():
    x = Tile(np.array([8, 4], dtype=np.int32))
    y = Tile(np.array([2, 2], dtype=np.int32))

    np.testing.assert_array_equal((x + y).data, [10, 6])
    np.testing.assert_array_equal((x - y).data, [6, 2])
    np.testing.assert_array_equal((10 - x).data, [2, 6])
    np.testing.assert_array_equal((x * y).data, [16, 8])
    np.testing.assert_array_equal((x / y).data, [4.0, 2.0])
    np.testing.assert_array_equal((16 / x).data, [2.0, 4.0])
    np.testing.assert_array_equal((x < y).data, [False, False])
    np.testing.assert_array_equal((x > y).data, [True, True])
    np.testing.assert_array_equal((x <= y).data, [False, False])
    np.testing.assert_array_equal((x >= y).data, [True, True])
    np.testing.assert_array_equal((x & y).data, [0, 0])
    np.testing.assert_array_equal((x | y).data, [10, 6])
    assert (x / y).dtype == ct.float32
    assert (x < y).dtype == ct.bool_

    # Test mixed int32/float32 promotion.
    mixed = Tile(np.array([1, 2], np.int32)) + Tile(np.array([0.5, 1.5], np.float32))
    assert mixed.dtype == ct.float32


def test_tile_operations():
    full = cutile_builder.full((2, 2), 3, ct.int32)
    reshaped = cutile_builder.reshape(full, (4,))
    converted = cutile_builder.astype(reshaped, ct.float32)
    total = cutile_builder.sum(full)

    lhs = Tile(np.array([[1, 2], [3, 4]], dtype=np.float32))
    rhs = Tile(np.array([[2, 0], [1, 2]], dtype=np.float32))
    product = cutile_builder.matmul(lhs, rhs)

    np.testing.assert_array_equal(full.data, [[3, 3], [3, 3]])
    np.testing.assert_array_equal(reshaped.data, [3, 3, 3, 3])
    assert converted.dtype == ct.float32
    assert total.data == 12
    np.testing.assert_array_equal(product.data, [[4, 4], [10, 8]])


def test_load_and_store():
    source = Array(np.arange(10, dtype=np.int32))
    first = cutile_builder.load(source, (0,), (4,))
    tail = cutile_builder.load(source, (2,), (4,), padding_mode=ct.PaddingMode.ZERO)

    output = np.full(10, -1, dtype=np.int32)
    cutile_builder.store(Array(output), (0,), first)
    cutile_builder.store(Array(output), (2,), tail)

    np.testing.assert_array_equal(first.data, [0, 1, 2, 3])
    np.testing.assert_array_equal(tail.data, [8, 9, 0, 0])
    np.testing.assert_array_equal(output, [0, 1, 2, 3, -1, -1, -1, -1, 8, 9])


def test_interpreted_kernel():
    @ct.kernel
    def add_kernel(a, b, out):
        block = ct.bid(0)
        x = ct.load(a, (block,), (4,), padding_mode=ct.PaddingMode.ZERO)
        y = ct.load(b, (block,), (4,), padding_mode=ct.PaddingMode.ZERO)
        ct.store(out, (block,), x + y)

    a = np.arange(10, dtype=np.int32)
    b = np.arange(10, dtype=np.int32) * 2
    output = np.empty_like(a)
    original_load = ct.load

    CuTileInterpretedFunction(add_kernel).run(a, b, output, grid=(3,))

    np.testing.assert_array_equal(output, a + b)
    assert ct.load is original_load
