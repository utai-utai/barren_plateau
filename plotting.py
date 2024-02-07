import csv
import json
from typing import List
from tqdm import trange
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
from barren_plateau import BP


class PLOTTING:
    def __init__(self, original_data: List = None, modified_data: List = None, saved_data: bool = False,
                 qubits: List[int] = None, layers: List[int] = None,
                 random_rotation_gate: List[str] = None, samples: int = 100, line_width: int = 3,
                 bar_width: float = 0.01, font_size: int = 30, legend_size: int = 15, label_size: int = 30,
                 absolute: bool = True):
        if qubits is None:
            qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18]
        if layers is None:
            layers = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500]
        if random_rotation_gate is None:
            random_rotation_gate = ['x', 'y', 'z']
        self.qubits = qubits
        self.layers = layers
        self.random_rotation_gate = random_rotation_gate
        self.samples = samples

        self.saved_data = saved_data
        self.original_data = original_data
        self.modified_data = modified_data
        self.original = BP(modify=False, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate)
        self.modified = BP(modify=True, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate)

        # plt.style.use(['science', 'no-latex'])
        plt.tick_params(width=2, labelsize=label_size)
        self.line_width = line_width
        self.bar_width = bar_width
        self.font_size = font_size
        self.legend_size = legend_size
        self.absolute = absolute

    @staticmethod
    def relist(data: List[dict], refer_key: str):
        values = []
        for i in data:
            values.append(i[refer_key])
        values = sorted(set(values))
        return values

    def use_saved_data(self):
        self.original_data = []
        self.modified_data = []
        with open("detail_data.csv", "r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row['qubit'] = int(row['qubit'])
                row['layer'] = int(row['layer'])
                row["paras"] = json.loads(row["paras"])
                row['outputs'] = json.loads(row['outputs'])
                row['gradients'] = json.loads(row['gradients'])
                row['variance'] = float(row['variance'])
                if row['modified'] == 'False':
                    self.original_data.append(row)
                else:
                    self.modified_data.append(row)
        self.qubits = self.relist(self.original_data, 'qubit')
        self.layers = self.relist(self.original_data, 'layer')
        return self.original_data, self.modified_data

    def transfer_detail_to_results(self):
        original, modified = self.use_saved_data()
        original_data = []
        modified_data = []
        for qubit in self.qubits:
            original_temp = []
            modified_temp = []
            for layer in self.layers:
                original_temp.extend(
                    i['variance'] for i in original if i.get('qubit') == qubit and i.get('layer') == layer)
                modified_temp.extend(
                    i['variance'] for i in modified if i.get('qubit') == qubit and i.get('layer') == layer)
                print(qubit, layer, original_temp)
            original_data.append(original_temp)
            modified_data.append(modified_temp)
        print(original_data)
        return original_data, modified_data

    def qubit_output(self, scatter: bool = True, bar: bool = True):
        if self.saved_data:
            original_data, modified_data = self.use_saved_data()
        elif self.original_data is None:
            original_data = self.original.get_detail()
            modified_data = self.modified.get_detail()
        else:
            original_data = self.original_data
            modified_data = self.modified_data
        if self.absolute:
            original_outputs = [[abs(j) for j in i['outputs']] for i in original_data]
            modified_outputs = [[abs(j) for j in i['outputs']] for i in modified_data]
        else:
            original_outputs = [i['outputs'] for i in original_data]
            modified_outputs = [i['outputs'] for i in modified_data]

        # Plot the scatter
        if scatter:
            original_paras = [i['paras'] for i in original_data]
            modified_paras = [i['paras'] for i in modified_data]
            for i in trange(len(original_paras)):
                plt.scatter(original_paras[i], original_outputs[i])
                plt.scatter(modified_paras[i], modified_outputs[i], color='red')
            plt.xlabel(r"$\theta_{1, 1}$", size=self.font_size)
            plt.ylabel(r"$\langle \psi |H| \psi \rangle$", size=self.font_size)
            plt.show()

        # Plot the bar
        if bar:
            if self.absolute:
                bins = np.arange(0, 1 + self.bar_width, self.bar_width)
            else:
                bins = np.arange(-1, 1 + self.bar_width, self.bar_width)
            original_qubit_index = [i['qubit'] for i in original_data]
            modified_qubit_index = [i['qubit'] for i in modified_data]
            original_hist = [0 for _ in range(len(original_outputs))]
            modified_hist = [0 for _ in range(len(modified_outputs))]
            for i in range(len(original_qubit_index)):
                original_hist[i], edges = np.histogram(original_outputs[i], bins=bins)
                plt.bar(edges[:-1], original_hist[i], width=np.diff(edges), edgecolor="white", align="edge",
                        alpha=0.6, label='original {} qubits'.format(original_qubit_index[i]))
            for i in range(len(modified_qubit_index)):
                modified_hist[i], edges = np.histogram(modified_outputs[i], bins=bins)
                plt.bar(edges[:-1], modified_hist[i], width=np.diff(edges), edgecolor="black", align="edge",
                        alpha=0.6, label='modified {} qubits'.format(modified_qubit_index[i]))
            plt.xlabel(r"$\langle \psi |H| \psi \rangle$", size=self.font_size)
            plt.ylabel('Frequency', size=self.font_size)
            plt.legend(fontsize=self.legend_size)
            plt.show()

    def qubit_gradient(self, scatter: bool = True, bar: bool = True):
        if self.saved_data:
            original_data, modified_data = self.use_saved_data()
        elif self.original_data is None:
            original_data = self.original.get_detail()
            modified_data = self.modified.get_detail()
        else:
            original_data = self.original_data
            modified_data = self.modified_data
        original_gradients = [i['gradients'] for i in original_data]
        modified_gradients = [i['gradients'] for i in modified_data]

        # Plot the scatter
        if scatter:
            original_paras = [i['paras'] for i in original_data]
            modified_paras = [i['paras'] for i in modified_data]
            for i in trange(len(original_paras)):
                plt.scatter(original_paras[i], original_gradients[i])
                plt.scatter(modified_paras[i], modified_gradients[i], color='red')
            plt.xlabel(r"$\theta_{1, 1}$", size=self.font_size)
            plt.ylabel("Gradients", size=self.font_size)
            plt.show()

        # Plot the bar
        if bar:
            bins = np.arange(-1, 1 + self.bar_width, self.bar_width)
            original_qubit_index = [i['qubit'] for i in original_data]
            modified_qubit_index = [i['qubit'] for i in modified_data]
            original_hist = [0 for _ in range(len(original_gradients))]
            modified_hist = [0 for _ in range(len(modified_gradients))]
            for i in trange(len(original_qubit_index)):
                original_hist[i], edges = np.histogram(original_gradients[i], bins=bins)
                plt.bar(edges[:-1], original_hist[i], width=np.diff(edges), edgecolor="white", align="edge",
                        alpha=0.6, label='original {} qubits'.format(original_qubit_index[i]))
            for i in trange(len(modified_qubit_index)):
                modified_hist[i], edges = np.histogram(modified_gradients[i], bins=bins)
                plt.bar(edges[:-1], modified_hist[i], width=np.diff(edges), edgecolor="black", align="edge",
                        alpha=0.6, label='modified {} qubits'.format(modified_qubit_index[i]))
            plt.xlabel('Gradient', size=self.font_size)
            plt.ylabel('Frequency', size=self.font_size)
            plt.legend(fontsize=self.legend_size)
            plt.show()

    def qubits_variance(self, refer_layer: int = 300):
        index = self.layers.index(refer_layer)
        if self.saved_data:
            original_data, modified_data = self.transfer_detail_to_results()
            print(original_data, modified_data)
        elif self.original_data is None:
            original_data = self.original.get_results()
            modified_data = self.modified.get_results()
        else:
            original_data = self.original_data
            modified_data = self.modified_data
        original_variance = [original_data[i][index] for i in range(len(self.qubits))]
        modified_variance = [modified_data[i][index] for i in range(len(self.qubits))]
        p = np.polyfit(self.qubits, np.log(original_variance), 1)  # original poly fitting
        q = np.polyfit(self.qubits, np.log(modified_variance), 1)  # modified poly fitting
        qubits = np.array(self.qubits)

        # Plot the straight line fit to the semi-logy
        plt.semilogy(qubits, original_variance, "o", label='Original')
        plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Original:Slope {:3.2f}".format(p[0]),
                     linewidth=self.line_width)
        plt.semilogy(qubits, modified_variance, 'o', label="Modified")
        plt.semilogy(qubits, np.exp(q[0] * qubits + q[1]), "o-.", label="Modified:Slope {:3.2f}".format(q[0]),
                     linewidth=self.line_width)
        plt.xlabel(r"N Qubits", fontsize=self.font_size)
        plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance", fontsize=self.font_size)
        plt.legend(fontsize=self.legend_size)
        plt.yscale('log')
        plt.show()

    def layers_variance(self):
        if self.saved_data:
            original_data, modified_data = self.transfer_detail_to_results()
        elif self.original_data is None:
            original_data = self.original.get_results()
            modified_data = self.modified.get_results()
        else:
            original_data = self.original_data
            modified_data = self.modified_data

        # Plot the line for each qubit
        for index, qubit in enumerate(self.qubits):
            plt.plot(self.layers, original_data[index], marker='*', label='original {} qubits'.format(qubit),
                     linewidth=self.line_width)
            plt.plot(self.layers, modified_data[index], marker='o', label='modified {} qubits'.format(qubit),
                     linewidth=self.line_width, color='red')
        plt.xlabel(r"Layers", fontsize=self.font_size)
        plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance", fontsize=self.font_size)
        plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=6, fontsize=self.legend_size)
        plt.yscale('log')
        plt.show()
