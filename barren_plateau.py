import csv
import json
from typing import List
from tqdm import trange
import numpy as np
import pennylane as qml


class BP:
    def __init__(self, modify: bool = False, qubits: List[int] = None, layers: List[int] = None,
                 random_rotation_gate: List[str] = None, samples: int = 100, result: bool = False):
        if qubits is None:
            qubits = [2, 4, 6]
        if layers is None:
            layers = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500]
        if random_rotation_gate is None:
            random_rotation_gate = ['x', 'y', 'z']
        self.modify = modify
        self.qubits = qubits
        self.layers = layers
        self.random_rotation_gate = random_rotation_gate
        self.samples = samples
        self.result = result
        self.detail_key = ['qubit', 'layer', 'paras', 'outputs', 'gradients', 'variance', 'modified']

    def rotation_gate(self, i: int, qubit: int, angle: float = None):
        gate = np.random.choice(self.random_rotation_gate)
        if angle is None:
            angle = 2 * np.pi * np.random.rand()
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

    def RPQCs(self, qubit: int, layer: int, theta: float):
        # initial layer
        if self.modify:
            for i in range(2 * qubit):
                qml.Hadamard(wires=i)
        else:
            for i in range(qubit):
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

    def main(self):
        detail = []
        results = []
        for qubit in self.qubits:
            gradients_variance = []
            for layer in self.layers:
                if self.modify:
                    dev = qml.device("default.qubit", wires=2 * qubit)
                else:
                    dev = qml.device("default.qubit", wires=qubit)

                @qml.qnode(dev, diff_method="parameter-shift")
                def circuit(theta):
                    return self.RPQCs(qubit, layer, theta)

                paras = []
                outputs = []
                gradients = []
                differential_circuit = qml.grad(circuit, argnum=0)
                for _ in trange(self.samples, desc='qubit={}, depth={}'.format(qubit, layer)):
                    para = 2 * np.pi * np.random.rand()
                    paras.append(para)
                    outputs.append(circuit(para))
                    gradients.append(differential_circuit(para))
                gradients_variance.append(np.var(gradients))
                detail.append(
                    {'qubit': qubit, 'layer': layer, 'paras': paras, 'outputs': outputs, 'gradients': gradients,
                     'variance': gradients_variance[-1], 'modified': self.modify})
            results.append(gradients_variance)
        return detail, results

    def get_detail(self):
        detail, _ = self.main()
        if self.result:
            print(detail)
        return detail

    def get_results(self):
        _, results = self.main()
        if self.result:
            print(results)
        return results

    def save_detail_results(self):
        detail, _ = self.main()
        with open("detail_data.csv", "a", newline="") as file:
            file.write('\n')
            detail_data = csv.DictWriter(file, fieldnames=self.detail_key)
            for row in detail:
                row['paras'] = json.dumps(row['paras'])
                row['outputs'] = json.dumps(row['outputs'])
                row['gradients'] = json.dumps(row['gradients'])
                detail_data.writerow(row)
