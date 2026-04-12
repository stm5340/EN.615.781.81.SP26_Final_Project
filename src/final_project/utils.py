import numpy as np
from numpy.typing import NDArray
from typing import List


def bit_error_rate(
    input_bits: NDArray[np.int64], output_bits: NDArray[np.int64]
) -> NDArray[np.int64]:
    """
    Computes the quantum bit error rate between two sets of bits
    """

    if np.size(input_bits) != np.size(output_bits):
        raise ValueError("input_bits and output_bits must be of the same size")

    if np.size(input_bits) == 0:
        return np.nan
    else:
        return np.sum(input_bits.flatten() != output_bits.flatten()) / np.size(
            input_bits
        )


def parity(bits: NDArray[np.int64]) -> np.int64:
    """
    Computes the parity of a set of bits
    """

    return int(np.bitwise_xor.reduce(bits))


def chunks(arr: NDArray[np.number], n: np.int64) -> List[NDArray[np.number]]:
    """
    Splits a N-element array into separate arrays each of size n where each entry
    is the corresponding index in the original array.

    Example: chunks(np.array([1, 2, 3, 4]), 2) -> [np.array([0, 1]), np.array([2, 3])]
    """

    return np.split(np.arange(arr.size), np.arange(n, len(arr), n))


def binary_search_parity_error(
    arr1: NDArray[np.int64], arr2: NDArray[np.int64], idx: NDArray[np.int64]
):
    """
    Performs a binary search on two arrays to locate a parity mismatch.
    """

    while idx.size > 1:
        mid = idx.size // 2  # Midpoint index
        left_idx = idx[:mid]  # Left side of array

        # Compare parity of blocks
        if parity(arr1[left_idx]) != parity(arr2[left_idx]):
            idx = left_idx
        else:
            idx = idx[mid:]

    return int(idx[0])


def binary_entropy(p: float) -> float:
    """
    Evaluates the binary entropy function
    """
    return (-p * np.log2(p) - (1 - p) * np.log2(1 - p))