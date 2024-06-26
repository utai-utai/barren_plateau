import numpy as np
import pennylane as qml
from tqdm import trange

from .setup_db import save_data


class BP:
    """
    This class is used to simulate the barren plateau phenomenon.
    """
    def __init__(self, modify: bool = False, qubits: list[int] = None, layers: list[int] = None, random_rotation_gate: list[str] = None, samples: int = 100, save: bool = False):
        """
        Initializes a new instance of the Class.

        Args:
            param modify: A boolean flag indicates whether to use the modified circuit.
            param qubits: A list of integers presents simulated qubits.
            param layers: A list of integers presents simulated layers.
            param random_rotation_gate: A list of 'x','y' and 'z'.
            param samples: An integer presents the repetitions of the circuits.
            param save: A boolean flag indicates whether to save the detail data.
        """
        try:
            if not isinstance(modify, bool):
                raise ValueError(f'modify={modify} must be a bool.')
            if qubits is None:
                qubits = [2, 4, 6]
                print('Test on {} qubits'.format(qubits))
            elif not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
                raise ValueError(f"qubits={qubits} must be a list of integers.")
            if layers is None:
                layers = [5, 10, 20, 50]
                print('Test on {} layers'.format(layers))
            elif not isinstance(layers, list) or not all(isinstance(_, int) for _ in layers):
                raise ValueError(f"layers={layers} must be a list of integers.")
            if random_rotation_gate is None:
                random_rotation_gate = ['x', 'y', 'z']
            elif not isinstance(random_rotation_gate, list) or not all(
                    isinstance(gate, str) for gate in random_rotation_gate):
                raise ValueError(f"random_rotation_gate={random_rotation_gate} must be ['x', 'y', 'z'].")
            if not isinstance(samples, int) or samples <= 0:
                raise ValueError(f"samples={samples} must be a positive integer.")
            if not isinstance(save, bool):
                raise ValueError(f"save={save} must be a bool.")
        except ValueError as e:
            print('Error initial parameter:', e)
            raise

        self.modify = modify
        self.qubits = qubits
        self.layers = layers
        self.random_rotation_gate = random_rotation_gate
        self.samples = samples
        self.save = save

    def rotation_gate(self, i: int, qubit: int, angle: float = None) -> qml.operation.Operation:
        """
        This function generates the rotation gate in each qubits and layers.

        Args:
            param i: An integer represents the current qubit.
            param qubit: An integer represents the number of the qubits.
            param angle: A float from 0 to 2π represents the rotation angle of the gate.

        Self:
            param modify, random_rotation_gate.

        Return:
            modified is True: CRX, CRY or CRZ.
            modified is False: RX, RY or RZ.
        """
        gate = np.random.choice(self.random_rotation_gate)
        if angle is None:
            angle = np.random.uniform(0, 2 * np.pi)
        if self.modify:
            if gate == 'x':
                return qml.CRX(angle, wires=[i + qubit, i])
            elif gate == 'y':
                return qml.CRY(angle, wires=[i + qubit, i])
            else:
                return qml.CRZ(angle, wires=[i + qubit, i])
        else:
            if gate == 'x':
                return qml.RX(angle, wires=i)
            elif gate == 'y':
                return qml.RY(angle, wires=i)
            else:
                return qml.RZ(angle, wires=i)

    def RPQCs(self, qubit: int, layer: int, theta: float) -> qml.measurements.ExpectationMP:
        """
        The main structure of the random parameterized quantum circuits(RPQCs).

        Args:
            param qubit: An integer presents simulated qubits.
            param layer: An integer presents simulated layers.
            param theta: A float from 0 to 2π represents the rotation angle for θ_{1,1}.

        Self:
            param modify.
            func rotation_gate().

        Return:
            The measurement of this circuit.
        """
        # initial layer
        for i in range((2 if self.modify else 1) * qubit):
            qml.Hadamard(wires=i)
        # parametrized layers
        for _ in range(layer):
            for i in range(qubit):
                if _ == 0 and i == 0:
                    self.rotation_gate(i, qubit, theta)
                else:
                    self.rotation_gate(i, qubit)
            for i in range(qubit - 1):
                qml.CZ(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def run(self) -> list[dict]:
        """
        This is the main process of this class, which is used to generate data.

        Self:
            param modify, qubits, layers, samples, save.
            func RPQCs(), save_detail_data().

        Return:
            param detail: A list of dictionary, which contains detail information.
        """
        detail = []
        for qubit in self.qubits:
            gradients_variance = []
            for layer in self.layers:
                dev = qml.device("default.qubit", wires=(2 if self.modify else 1) * qubit)

                @qml.qnode(dev, diff_method="parameter-shift")
                def circuit(theta):
                    return self.RPQCs(qubit, layer, theta)

                paras = []
                gradients = []
                differential_circuit = qml.grad(circuit, argnum=0)
                for _ in trange(self.samples, desc='qubit={}, depth={}'.format(qubit, layer)):
                    para = np.random.uniform(0, 2 * np.pi)
                    paras.append(para)
                    gradients.append(differential_circuit(para))
                gradients_variance.append(np.var(gradients))
                detail.append({'modified': self.modify, 'qubit': qubit, 'layer': layer, 'paras': paras, 'gradients': gradients, 'variance': gradients_variance[-1]})
                if self.save:
                    save_data(detail[-1], 'single')
        return detail

    def variance(self) -> list[list]:
        """
           Return the results. If result=True(default), then print the variance data.

           Self:
                param qubits, layers.
                func run.

           Return:
                param results: A list of lists, which contain the variance of the gradients.
                len(result)=len(qubits) and len(result[0])=len(layers).
       """
        results = []
        for qubit in self.qubits:
            result = []
            for layer in self.layers:
                result.extend(i['variance'] for i in self.run() if i.get('qubit') == qubit and i.get('layer') == layer)
            results.append(result)
        return results
