import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import trange

from .barren_plateau import BP
from .setup_db import read_data


class PLOTTING:
    """
    This class is used to plot the results of the barren plateau.
    """
    def __init__(self, original_data: list = None, modified_data: list = None, saved_data: bool = False, qubits: list[int] = None, layers: list[int] = None, selected_qubits: list[int] = None, selected_layers: list[int] = None, random_rotation_gate: list[str] = None, samples: int = 100, line_width: int = 3, bar_width: float = 0.01, font_size: int = 30, legend_size: int = 25, label_size: int = 30, absolute: bool = True) -> None:
        """
        Initializes a new instance of the Class.

        Args:
            param original_data: A list of dictionaries contains barren plateaus data
            param modified_data: A list of dictionaries contains modified circuit data
            param saved_data: a boolean flag indicates whether to use the data that saved in 'detail_data.db'.
            param qubits: A list of integers presents simulated qubits.
            param layers: A list of integers presents simulated layers.
            param random_rotation_gate: A list of 'x','y' and 'z'.
            param samples: An integer presents the repetitions of the circuits.
            param line_width: An integer presents the line width.
            param bar_width: A float presents the bar width.
            param font_size: An integer presents the font size.
            param legend_size: An integer presents the legend size.
            param label_size: An integer presents the label size.
            param absolute: A boolean flag indicates whether to use absolute data of outputs.

        Self:
            param original: A class uses original circuit .
            param modified: A class uses modified circuit.
        """
        try:
            if not isinstance(saved_data, bool):
                raise ValueError(f"saved_data={saved_data} must be a bool.")
            if saved_data is False:
                if not isinstance(original_data, list):
                    raise ValueError(f"original_data={original_data} must be a list.")
                if not isinstance(modified_data, list):
                    raise ValueError(f"modified_data={modified_data} must be a list.")
            if qubits is None:
                qubits = [2, 4, 6]
            elif not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
                raise ValueError(f"qubits={qubits} must be a list of integers.")
            if layers is None:
                layers = [5, 10, 20, 50]
            elif not isinstance(layers, list) or not all(isinstance(_, int) for _ in layers):
                raise ValueError(f"layers={layers} must be a list of integers.")
            if selected_qubits is None:
                selected_qubits = [2, 4, 6]
            elif not isinstance(selected_qubits, list) or not all(isinstance(q, int) for q in selected_qubits):
                raise ValueError(f"selected_qubits={selected_qubits} must be a list of integers.")
            if selected_layers is None:
                selected_layers = [5, 10, 20, 50]
            elif not isinstance(selected_layers, list) or not all(isinstance(_, int) for _ in selected_layers):
                raise ValueError(f"selected_layers={selected_layers} must be a list of integers.")
            if random_rotation_gate is None:
                random_rotation_gate = ['x', 'y', 'z']
            elif not isinstance(random_rotation_gate, list) or not all(
                    isinstance(gate, str) for gate in random_rotation_gate):
                raise ValueError(f"random_rotation_gate={random_rotation_gate} must be ['x', 'y', 'z'].")
            if not isinstance(samples, int) or samples <= 0:
                raise ValueError(f"samples={samples} must be a positive integer.")
            if not isinstance(line_width, int) or line_width <= 0:
                raise ValueError(f"line_width{line_width} must be a positive integer.")
            if not isinstance(bar_width, float) or bar_width <= 0:
                raise ValueError(f"bar_width={bar_width} must be a positive float.")
            if not isinstance(font_size, int) or font_size <= 0:
                raise ValueError(f"font_size={font_size} must be a positive integer.")
            if not isinstance(legend_size, int) or legend_size <= 0:
                raise ValueError(f"legend_size={legend_size} must be a positive integer.")
            if not isinstance(label_size, int) or label_size <= 0:
                raise ValueError(f"label_size={label_size} must be a positive integer.")
            if not isinstance(absolute, bool):
                raise ValueError(f"absolute={absolute} must be a bool.")
        except ValueError as e:
            print("Error initial parameter:", e)
            raise

        self.qubits = qubits
        self.layers = layers
        self.random_rotation_gate = random_rotation_gate
        self.samples = samples

        self.saved_data = saved_data
        self.original_data = original_data
        self.modified_data = modified_data
        self.original = BP(modify=False, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate)
        self.modified = BP(modify=True, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate)

        self.selected_qubits = selected_qubits
        self.selected_layers = selected_layers
        self.line_width = line_width
        self.bar_width = bar_width
        self.font_size = font_size
        self.legend_size = legend_size
        self.label_size = label_size
        self.absolute = absolute

    def relist(self, refer_key: str):
        """
        This function is used to count the occurrence of qubits and layers.

        Args:
            param data: A list of dictionaries that contains the key:qubits and key:layers.
            param refer_key: A string of qubits or layers.

        Return:
            param values: A list of integers, which is arranged in ascending order.
        """
        try:
            if refer_key != 'qubit' and refer_key != 'layer':
                raise ValueError(f'refer_key={refer_key} must be "qubit" or "layer".')
        except ValueError as e:
            print('Error parameter:', e)
            raise
        values = []
        for i in self.original_data:
            values.append(i[refer_key])
        values = sorted(set(values))
        return values

    def use_saved_data(self):
        """
        This function is used to transfer the data in 'data.db' to a list of dictionaries.
        It will transfer the para{qubits} and para{layers} that base on the 'detail_data.db'.

        Self:
            param original_data, modified_data, qubits, layers.
            func relist().

        Return:
            param original_data: A list of dictionaries, which contains the detail data about original circuit.
            param modified_data: A list of dictionaries, which contains the detail data about modified circuit.
        """
        self.original_data, self.modified_data = read_data()
        self.qubits = self.relist('qubit')
        self.layers = self.relist('layer')
        return self.original_data, self.modified_data

    def transfer_detail_to_results(self):
        """
        This function is used to transfer the detail data to a list of variance.

        Self:
            param qubits, layers.
            func use_saved_data().

        Return:
            param original_data: A list of dictionaries, which contains the variance about original circuit.
            param modified_data: A list of dictionaries, which contains the variance about modified circuit.
        """
        original, modified = self.use_saved_data()
        original_data = []
        modified_data = []
        for qubit in self.qubits:
            original_temp = []
            modified_temp = []
            for layer in self.layers:
                original_temp.extend(i['variance'] for i in original if i.get('qubit') == qubit and i.get('layer') == layer)
                modified_temp.extend(i['variance'] for i in modified if i.get('qubit') == qubit and i.get('layer') == layer)
            original_data.append(original_temp)
            modified_data.append(modified_temp)
        return original_data, modified_data

    def select_required_data(self, datas: list[dict]):
        """
        This function is used to select the data by using required qubits and layers.

        Arg:
            param data: A list of dictionaries contains all the datas.

        Self:
            param selected_qubits, selected_layers.

        Return:
             selected_data: A list of dictionaries contains selected datas.
        """
        selected_datas = []
        for data in datas:
            if data['qubit'] in self.selected_qubits and data['layer'] in self.selected_layers:
                selected_datas.append(data)
        return selected_datas

    def qubit_output(self, scatter: bool = True, bar: bool = True):
        """
        This function plots the relationship between qubits and outputs.
        It will order 3 different mode.
            1. If para{saved_data} is True, it will use the saved data.
            2. If para{original_data} or para{modified_data} is not None, it will use the offered data.
            3. Else, it will use the data by class BP

        Arg:
            param scatter: A boolean flag indicates whether to plot scatter.
            param bar: A boolean flag indicates whether to plot bar.

        Self:
            param saved_data, qubits, layers, absolute.
            class original(BP), modified(BP)
            plot param font_size, bar_width, legend_size
            func use_saved_data().

        Return:
            plt.show()
        """
        try:
            if not isinstance(scatter, bool):
                raise ValueError(f'scatter={scatter} must be a bool')
            if not isinstance(bar, bool):
                raise ValueError(f'bar={bar} must be a bool')
        except ValueError as e:
            print('Error parameter:', e)
            raise
        if self.saved_data:
            original_data, modified_data = self.use_saved_data()
        elif self.original_data is None:
            original_data = self.original.run()
            modified_data = self.modified.run()
        else:
            original_data = self.original_data
            modified_data = self.modified_data
        original_data = self.select_required_data(original_data)
        modified_data = self.select_required_data(modified_data)
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
                plt.bar(edges[:-1], original_hist[i], width=np.diff(edges), edgecolor="white", align="edge", alpha=0.6, label='original {} qubits'.format(original_qubit_index[i]))
            for i in range(len(modified_qubit_index)):
                modified_hist[i], edges = np.histogram(modified_outputs[i], bins=bins)
                plt.bar(edges[:-1], modified_hist[i], width=np.diff(edges), edgecolor="black", align="edge", alpha=0.6, label='modified {} qubits'.format(modified_qubit_index[i]))
            plt.xlabel(r"$\langle \psi |H| \psi \rangle$", size=self.font_size)
            plt.ylabel('Frequency', size=self.font_size)
            plt.legend(fontsize=self.legend_size)
            plt.show()

    def qubit_gradient(self, scatter: bool = True, bar: bool = True):
        """
        This function plots the relationship between qubits and gradients.
        It will order 3 different mode.
            1. If param{saved_data} is True, it will use the saved data.
            2. If param{original_data} or param{modified_data} is not None, it will use the offered data.
            3. Else, it will use the data by class BP

        Arg:
            param scatter: A boolean flag indicates whether to plot scatter.
            param bar: A boolean flag indicates whether to plot bar.

        Self:
            param saved_data, qubits, layers.
            class original(BP), modified(BP).
            plot param font_size, bar_width, legend_size.
            func use_saved_data().

        Return:
            plt.show()
        """
        try:
            if not isinstance(scatter, bool):
                raise ValueError(f'scatter={scatter} must be a bool.')
            if not isinstance(bar, bool):
                raise ValueError(f'bar={bar} must be a bool.')
        except ValueError as e:
            print('Error parameter:', e)
            raise
        if self.saved_data:
            original_data, modified_data = self.use_saved_data()
        elif self.original_data is None:
            original_data = self.original.run()
            modified_data = self.modified.run()
        else:
            original_data = self.original_data
            modified_data = self.modified_data
        original_data = self.select_required_data(original_data)
        modified_data = self.select_required_data(modified_data)
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
                plt.bar(edges[:-1], original_hist[i], width=np.diff(edges), edgecolor="white", align="edge", alpha=0.6, label='original {} qubits'.format(original_qubit_index[i]))
            for i in trange(len(modified_qubit_index)):
                modified_hist[i], edges = np.histogram(modified_gradients[i], bins=bins)
                plt.bar(edges[:-1], modified_hist[i], width=np.diff(edges), edgecolor="black", align="edge", alpha=0.6, label='modified {} qubits'.format(modified_qubit_index[i]))
            plt.xlabel('Gradient', size=self.font_size)
            plt.ylabel('Frequency', size=self.font_size)
            plt.legend(fontsize=self.legend_size)
            plt.show()

    def qubits_variance(self, refer_layer: int = 500):
        """
        This function plots the relationship between qubits and the variance of the gradient.
        It will order 3 different mode.
            1. If para{saved_data} is True, it will use the saved data.
            2. If para{original_data} or para{modified_data} is not None, it will use the offered data.
            3. Else, it will use the data by class BP

        Arg:
            param refer_layer: An integer presents the referred layer.

        Self:
            param saved_data, qubits, layers.
            class original(BP), modified(BP).
            plot param line_width, font_size, legend_size.
            func use_saved_data(), transfer_detail_to_results().

        Return:
            plt.show()
        """
        if self.saved_data:
            original_data, modified_data = self.transfer_detail_to_results()
        elif self.original_data is None:
            original_data = self.original.variance()
            modified_data = self.modified.variance()
        else:
            original_data = self.original_data
            modified_data = self.modified_data
        try:
            if not isinstance(refer_layer, int) or refer_layer <= 0:
                raise ValueError(f'refer_layer={refer_layer} must be a positive integer.')
            if refer_layer not in self.layers:
                raise ValueError(f'refer_layer={refer_layer} can not be found in layers={self.layers}')
        except ValueError as e:
            print('Error parameter:', e)
            raise
        index = self.layers.index(refer_layer)
        original_variance = [original_data[i][index] for i in range(len(self.qubits))]
        modified_variance = [modified_data[i][index] for i in range(len(self.qubits))]
        p = np.polyfit(self.qubits, np.log(original_variance), 1)  # original poly fitting
        q = np.polyfit(self.qubits, np.log(modified_variance), 1)  # modified poly fitting
        qubits = np.array(self.qubits)

        # Plot the straight line fit to the semi-logy
        plt.figure(figsize=(16, 9))
        plt.semilogy(qubits, original_variance, "o", label='Barren Plateau')
        plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Barren Plateau:Slope {:3.2f}".format(p[0]), linewidth=self.line_width)
        plt.semilogy(qubits, modified_variance, 'o', label="Modified")
        plt.semilogy(qubits, np.exp(q[0] * qubits + q[1]), "o-.", label="Modified:Slope {:3.2f}".format(q[0]), linewidth=self.line_width)
        plt.xlabel(r"N Qubits", fontsize=self.font_size)
        plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance", fontsize=self.font_size, fontweight='bold')
        plt.legend(fontsize=self.legend_size)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize=self.label_size, width=3)
        plt.show()

    def layers_variance(self):
        """
        This function plots the relationship between layers and the variance of the gradient.
        It will order 3 different mode.
            1. If para{saved_data} is True, it will use the saved data.
            2. If para{original_data} or para{modified_data} is not None, it will use the offered data.
            3. Else, it will use the data by class BP

        Self:
            param saved_data, qubits, layers.
            class original(BP), modified(BP).
            plot param line_width, font_size, legend_size.
            func use_saved_data(), transfer_detail_to_results().

        Return:
            plt.show()
        """
        if self.saved_data:
            original_data, modified_data = self.transfer_detail_to_results()
        elif self.original_data is None:
            original_data = self.original.variance()
            modified_data = self.modified.variance()
        else:
            original_data = self.original_data
            modified_data = self.modified_data

        # Plot the line for each qubit
        plt.figure(figsize=(16, 9))
        for index, qubit in enumerate(self.qubits):
            if index == 0:
                plt.plot(self.layers, original_data[index], marker='*', label='Barren Plateau', linewidth=self.line_width, color='green')
            else:
                plt.plot(self.layers, original_data[index], marker='*', linewidth=self.line_width, color='green', alpha=0.91-0.07*index)
        for index, qubit in enumerate(self.qubits):
            if index == 0:
                plt.plot(self.layers, modified_data[index], marker='o', label='Modified', linewidth=self.line_width, color='red')
            else:
                plt.plot(self.layers, modified_data[index], marker='o', linewidth=self.line_width, color='red', alpha=0.91-0.07*index)
        plt.xlabel(r"Layers", fontsize=self.font_size, fontweight='bold')
        plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance", fontsize=self.font_size, fontweight='bold')
        plt.legend(fontsize=self.legend_size)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize=self.label_size, width=3)
        plt.show()


def plot_simple_datas(original: list = None, modified: list = None, name: str = None, scatter: bool = True, bar: bool = True):
    """
    This function plots simple data in scatter and bar.

    Arg:
        param data_0 and data_1: A list * 2
        param scatter: A boolean flag indicates whether to plot scatter.
        param bar: A boolean flag indicates whether to plot bar.

    Return:
        plt.show()
    """
    try:
        if not isinstance(original, list):
            raise ValueError(f'original={original} must be a list.')
        if not isinstance(modified, list):
            raise ValueError(f'modified={modified} must be a list.')
        if not isinstance(name, str):
            raise ValueError(f'name={name} must be a string.')
        if not isinstance(scatter, bool):
            raise ValueError(f'scatter={scatter} must be a bool.')
        if not isinstance(bar, bool):
            raise ValueError(f'bar={bar} must be a bool.')
    except ValueError as e:
        print('Error parameter:', e)
        raise

    line_width: int = 3
    bar_width: float = 0.01
    font_size: int = 30
    legend_size: int = 25
    label_size: int = 30

    if scatter:
        steps = len(original)
        plt.scatter(range(steps), original, marker='*', color='green', label='Barren Plateau')
        plt.scatter(range(steps), modified, marker='.', color='red', label='Modified')
        plt.xlabel("Epochs", fontsize=font_size)
        plt.ylabel(rf"{name}", fontsize=font_size)
        plt.legend(fontsize=legend_size)
        plt.tick_params(width=2, labelsize=label_size)
        plt.show()

    if bar:
        # Plot the bar and KDE carve of original data
        counts_orig, bin_edges_orig = np.histogram(original, bins=np.arange(-1, 1.01, bar_width), density=False)
        bin_centers_orig = (bin_edges_orig[:-1] + bin_edges_orig[1:]) / 2
        kde_orig = gaussian_kde(original)
        kde_values_orig = kde_orig(np.arange(-1, 1 + bar_width, bar_width))
        plt.bar(bin_centers_orig, counts_orig / len(original), width=bar_width, alpha=0.6, color='blue')
        plt.plot(np.arange(-1, 1 + bar_width, bar_width), kde_values_orig / sum(kde_values_orig), label='Unitary 2-design', color='green', linewidth=line_width, alpha=0.8)

        # Plot the bar and KDE carve of modified data
        counts_mod, bin_edges_mod = np.histogram(modified, bins=np.arange(-1, 1.01, bar_width), density=False)
        bin_centers_mod = (bin_edges_mod[:-1] + bin_edges_mod[1:]) / 2
        kde_mod = gaussian_kde(modified)
        kde_values_mod = kde_mod(np.arange(-1, 1 + bar_width, bar_width))
        plt.bar(bin_centers_mod, counts_mod / len(modified), width=bar_width, alpha=0.6, color='orange')
        plt.plot(np.arange(-1, 1 + bar_width, bar_width), kde_values_mod / sum(kde_values_mod), label='Proposed Structure', color='red', linewidth=line_width, alpha=0.8)

        plt.xlabel(rf"{name}", fontsize=font_size)
        plt.ylabel('Frequency', fontsize=font_size)
        plt.legend(fontsize=legend_size)
        plt.tick_params(width=2, labelsize=label_size)
        plt.show()
