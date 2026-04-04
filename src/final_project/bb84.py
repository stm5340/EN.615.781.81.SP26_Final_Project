from copy import deepcopy
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, AerJob
from itertools import compress

SEED = None  # Overwrite with fixed value for reproducible results
rng = np.random.default_rng(SEED)

BASIS_CHOICES = ["Z", "X"]
BIT_CHOICES = [0, 1]


class BB84(QuantumCircuit):
    """
    Class that implements a BB84 circuit with no eavesdropping using the qiskit library's QuantumCircuit class
    """

    input_bit: int = None
    alice_basis: str = None
    bob_basis: str = None
    eve_basis: str = None

    def __init__(
        self, input_bit: int, alice_basis: str, bob_basis: str, eve_basis: str
    ):

        super().__init__(1, 1, name="BB84")

        if input_bit != 0 and input_bit != 1:
            raise ValueError("Input bit must take the classical state of '0' or '1'")

        self.initialize(str(input_bit))

        if alice_basis == "Z":  # Alice chooses to prepare the message in the Z-basis
            self.id(0)
        elif alice_basis == "X":  # Alice chooses to prepare the message in the X-basis
            self.h(0)
        else:
            raise ValueError("Alice must choose an input basis of 'Z' or 'X'")

        if eve_basis == "Z":  # Eve chooses to prepare the message in the Z-basis
            self.id(0)
        elif eve_basis == "X":  # Eve chooses to prepare the message in the X-basis
            self.h(0)
        else:
            raise ValueError("Eve must choose an input basis of 'Z' or 'X'")

        self.measure(0, 0)  # Eve measures the messages ("intercept")

        cbit = self.clbits[
            0
        ]  # Classical bit value of zeroth index qubit post-measurement
        self.reset(
            0
        )  # Reset the zeroth index qubit since Eve has to resend the message

        if cbit == 0:
            # Eve's basis choice didn't yield a positive measurement
            # She can't recreate the supposed initial state since she doesn't
            # have enough information.
            self.initialize("0")
        else:
            # Eve's basis choice yielded a positive measurement
            self.initialize("1")

        if eve_basis == "Z":  # Eve chooses to resend the message in the Z-basis
            self.id(0)
        elif eve_basis == "X":  # Eve chooses to resend the message in the X-basis
            self.h(0)
        else:
            raise ValueError("Eve must choose an input basis of 'Z' or 'X'")

        if bob_basis == "Z":  # Bob chooses to measure the message in the Z-basis
            self.id(0)
        elif bob_basis == "X":  # Bob chooses to measure the message in the X-basis
            self.h(0)
        else:
            raise ValueError("Bob must choose an input basis of 'Z' or 'X'")

        self.input_bit = input_bit
        self.alice_basis = alice_basis
        self.bob_basis = bob_basis
        self.eve_basis = eve_basis

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


def create_simulation_space(n_messages: int) -> SimulationInputs:
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
            BB84(input_bits[ii], alice_bases[ii], bob_bases[ii], eve_bases[ii])
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


def bit_error_rate(input_bits: NDArray[np.int64], output_bits: NDArray[np.int64]):
    """
    Computes the quantum bit error rate between two sets of bits
    """

    if np.size(input_bits) != np.size(output_bits):
        raise ValueError("input_bits and output_bits must be of the same size")

    return np.sum(input_bits.flatten() != output_bits.flatten()) / np.size(input_bits)


def main() -> None:
    sim_inputs = create_simulation_space(n_messages=128)
    output_bits = simulate_communication(sim_inputs=sim_inputs)
    output_bits = output_bits[sim_inputs.alice_bases == sim_inputs.bob_bases]
    sim_inputs.discard_data()  # Remove inputs where Alice and Bob don't have the same basis
    qber = bit_error_rate(input_bits=sim_inputs.input_bits, output_bits=output_bits)
    print(f"Computed quantum bit error rate in simulation: {qber}")


if __name__ == "__main__":

    main()
