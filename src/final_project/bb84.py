from copy import deepcopy
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Clbit
from qiskit_aer import AerSimulator, AerJob
from itertools import compress
from utils import (
    bit_error_rate,
    parity,
    binary_search_parity_error,
    chunks,
    binary_entropy,
)

SEED = None  # Overwrite with fixed value for reproducible results
rng = np.random.default_rng(SEED)

BASIS_CHOICES = ["Z", "X"]
BIT_CHOICES = [0, 1]


class BB84(QuantumCircuit):
    """
    Class that implements a BB84 circuit with configurable eavesdropping using the qiskit library's QuantumCircuit class
    """

    input_bit: int = None
    alice_basis: str = None
    bob_basis: str = None
    eve_basis: Optional[str] = None

    def __init__(
        self,
        input_bit: int,
        alice_basis: str,
        bob_basis: str,
        eve_basis: str,
        eve_present: bool = False,
    ):

        super().__init__(1, 1, name="BB84")

        if input_bit != 0 and input_bit != 1:
            raise ValueError("Input bit must take the classical state of '0' or '1'")

        if input_bit == 0:
            # Qiskit initializes qubit in |0> state, no need to do anything
            pass
        else:
            # Apply a bit flip to get the state |1>
            self.x(0)

        self.input_bit = input_bit

        if alice_basis == "Z":  # Alice chooses to prepare the message in the Z-basis
            self.id(0)
        elif alice_basis == "X":  # Alice chooses to prepare the message in the X-basis
            self.h(0)
        else:
            raise ValueError("Alice must choose an input basis of 'Z' or 'X'")

        self.alice_basis = alice_basis

        if eve_present:
            if eve_basis == "Z":  # Eve chooses to prepare the message in the Z-basis
                self.id(0)
            elif eve_basis == "X":  # Eve chooses to prepare the message in the X-basis
                self.h(0)
            else:
                raise ValueError("Eve must choose an input basis of 'Z' or 'X'")

            self.measure(0, 0)  # Eve measures the messages ("intercept")

            cbit: Clbit = self.clbits[
                0
            ]  # Classical bit value of zeroth index qubit post-measurement
            self.reset(
                0
            )  # Reset the zeroth index qubit since Eve has to resend the message

            with self.if_test((cbit, 1)) as else_:
                self.x(0)
            with else_:
                self.id(0)

            if eve_basis == "Z":  # Eve chooses to resend the message in the Z-basis
                self.id(0)
            elif eve_basis == "X":  # Eve chooses to resend the message in the X-basis
                self.h(0)
            else:
                raise ValueError("Eve must choose an input basis of 'Z' or 'X'")

            self.eve_basis = eve_basis

        if bob_basis == "Z":  # Bob chooses to measure the message in the Z-basis
            self.id(0)
        elif bob_basis == "X":  # Bob chooses to measure the message in the X-basis
            self.h(0)
        else:
            raise ValueError("Bob must choose an input basis of 'Z' or 'X'")

        self.bob_basis = bob_basis

        self.measure(0, 0)  # Measure the first register (zeroth index)


@dataclass
class SimulationInputs:
    """
    Data class containing all chosen inputs by Alice and Bob for each message in the simulation
    """

    input_bits: NDArray[np.int64]
    alice_bases: NDArray[np.str_]
    bob_bases: NDArray[np.str_]
    eve_bases: NDArray[np.str_]
    circuits: List[BB84]

    def discard_data(self) -> Tuple[object, object]:
        """
        Remove data where Alice and Bob don't use the same basis
        """
        raw = deepcopy(self)  # Provide copy of unaltered data if desired
        mask = self.alice_bases == self.bob_bases
        self.input_bits = self.input_bits[mask]
        self.alice_bases = self.alice_bases[mask]
        self.bob_bases = self.bob_bases[mask]
        self.eve_bases = self.eve_bases[mask]
        self.circuits = list(compress(self.circuits, mask))
        return self, raw


def create_simulation_space(
    n_messages: int, eve_present: bool = False
) -> SimulationInputs:
    """
    Creates a set of BB84 circuits where each input parameter is drawn from a uniform distribution
    over the possible range of values
    """
    alice_bases = rng.choice(BASIS_CHOICES, size=n_messages)
    bob_bases = rng.choice(BASIS_CHOICES, size=n_messages)
    eve_bases = rng.choice(BASIS_CHOICES, size=n_messages)
    input_bits = rng.choice(BIT_CHOICES, size=n_messages)

    sim_inputs = SimulationInputs(
        input_bits,
        alice_bases,
        bob_bases,
        eve_bases,
        [
            BB84(
                input_bits[ii],
                alice_bases[ii],
                bob_bases[ii],
                eve_bases[ii],
                eve_present,
            )
            for ii in range(n_messages)
        ],
    )

    return sim_inputs


def simulate_communication(sim_inputs: SimulationInputs) -> NDArray[np.int64]:
    """
    Given a set of simulation inputs, simulate communication between Alice and Bob, and record the bit Bob measure's
    """
    backend = AerSimulator()
    circuits: List[BB84] = sim_inputs.circuits
    num_messages = len(circuits)
    results: AerJob = backend.run(
        transpile(circuits, backend), shots=1, memory=True
    ).result()
    ouput_bits = []
    for i in range(num_messages):
        ouput_bits.append(results.get_memory(i)[0])

    return np.asarray(ouput_bits, dtype=np.int64)


def cascade_protocol(
    alice_key: NDArray[np.int64],
    bob_key: NDArray[np.int64],
    chunk_size: int,
    idx_perm=None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Implementation of Cascade Protocol
    """

    if idx_perm is None:
        idx_perm = np.arange(alice_key.size)

    # Re-organize bits according to input permutation
    alice_perm = alice_key[idx_perm]
    bob_perm = bob_key[idx_perm]

    idx_err = np.array([], dtype=np.int64)
    idx_blocks = chunks(np.arange(alice_key.size, dtype=np.int64), chunk_size)
    for idx in idx_blocks:

        # Compute parity of sub-set of bits
        alice_parity = parity(alice_perm[idx])
        bob_parity = parity(bob_perm[idx])

        if alice_parity != bob_parity:
            # Find the error and correct the parity
            err_loc = binary_search_parity_error(alice_perm, bob_perm, idx)
            bob_perm[err_loc] ^= 1
            idx_err = np.append(idx_err, err_loc)

    # Correct Bob's bits
    bob_corrected = bob_key
    for ii, idx in enumerate(idx_perm):
        bob_corrected[idx] = bob_perm[ii]

    idx_err = idx_perm[idx_err]  # Indices needing correction
    eve_information = len(idx_blocks)  # Information gained by Eve

    return bob_corrected, idx_err, eve_information


def information_reconciliation(
    alice_key: NDArray[np.int64],
    bob_key: NDArray[np.int64],
    qber_estimate: float,
    n_passes: int = 4,
    seed: int = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Perform information reconciliation by performing the Cascade protocol
    `n_passes` times
    """

    rng = np.random.default_rng(seed)
    n = len(alice_key)

    # First block-size is inversely proportional to QBER
    block_size = int(1 / qber_estimate)
    eve_information = 0  # Accumulate additional information gained by Eve
    for _ in range(n_passes):
        perm = np.arange(n)
        rng.shuffle(perm)

        bob_key, _, eve_bits_acc = cascade_protocol(
            alice_key, bob_key, block_size, perm
        )
        eve_information += eve_bits_acc

        block_size *= 2  # Increase block size for next iteration

    return bob_key, eve_information

def privacy_amplification(
    reconciled_key: NDArray[np.int64],
    key_length: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Privacy amplification using a random binary matrix over GF(2).
    """

    hash_matrix = rng.integers(
        0, 2, size=(key_length, reconciled_key.size), dtype=np.int64
    )
    final_key = (hash_matrix @ reconciled_key) % 2

    return final_key.astype(np.int64), hash_matrix


def simulate_no_eve_present() -> None:
    sim_inputs = create_simulation_space(n_messages=2**12)
    output_bits = simulate_communication(sim_inputs=sim_inputs)
    output_bits = output_bits[sim_inputs.alice_bases == sim_inputs.bob_bases]
    sim_inputs.discard_data()  # Remove inputs where Alice and Bob don't have the same basis
    qber = bit_error_rate(input_bits=sim_inputs.input_bits, output_bits=output_bits)
    print(f"Computed quantum bit error rate in simulation: {qber}")


def simulate_eve_present() -> None:

    qber: List = []
    sim_inputs = create_simulation_space(n_messages=2**12, eve_present=True)
    output_bits = simulate_communication(sim_inputs=sim_inputs)
    output_bits = output_bits[sim_inputs.alice_bases == sim_inputs.bob_bases]
    sim_inputs.discard_data()  # Remove inputs where Alice and Bob don't have the same basis
    qber.append(
        bit_error_rate(input_bits=sim_inputs.input_bits, output_bits=output_bits)
    )
    print(f"Computed quantum bit error rate in simulation: {np.mean(qber)}")


def simulate_information_reconcilation() -> None:
    sim_inputs = create_simulation_space(n_messages=2**12, eve_present=True)
    output_bits = simulate_communication(sim_inputs=sim_inputs)
    output_bits = output_bits[sim_inputs.alice_bases == sim_inputs.bob_bases]
    sim_inputs.discard_data()  # Remove inputs where Alice and Bob don't have the same basis
    estimated_qber = bit_error_rate(
        input_bits=sim_inputs.input_bits[0:1000], output_bits=output_bits[0:1000]
    )  # Estimate QBER using subset of remaining message
    (corrected_output_bits, eve_info) = information_reconciliation(
        alice_key=sim_inputs.input_bits[1000:],
        bob_key=output_bits[1000:],
        qber_estimate=estimated_qber,
        n_passes=5,
        seed=10,
    )

    print(
        f"Information gained by Eve through information reconciliation: {eve_info} \n"
    )

    print(f"Estimated QBER: {estimated_qber} \n")

    actual_qber = bit_error_rate(
        input_bits=sim_inputs.input_bits, output_bits=output_bits
    )

    print(f"Actual QBER: {actual_qber} \n")

    corrected_qber = bit_error_rate(
        input_bits=sim_inputs.input_bits[1000:], output_bits=corrected_output_bits
    )
    print(f"Corrected QBER {corrected_qber} \n")


if __name__ == "__main__":
    simulate_information_reconcilation()
