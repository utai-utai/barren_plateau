import json
import sqlite3
import numpy as np
from tqdm import trange
import scienceplots
import matplotlib.pyplot as plt
from .barren_plateau import BP


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
                raise ValueError("para:{saved_data} must be a bool.")
            if saved_data is False:
                if not isinstance(original_data, list):
                    raise ValueError("para:{original_data} must be a list.")
                if not isinstance(modified_data, list):
                    raise ValueError("para:{modified_data} must be a list.")
            if qubits is None:
                qubits = [2, 4, 6]
            elif not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
                raise ValueError("para:{qubits} must be a list of integers.")
            if layers is None:
                layers = [5, 10, 20, 50]
            elif not isinstance(layers, list) or not all(isinstance(_, int) for _ in layers):
                raise ValueError("para:{layers} must be a list of integers.")
            if selected_qubits is None:
                selected_qubits = [2, 4, 6]
            elif not isinstance(selected_qubits, list) or not all(isinstance(q, int) for q in selected_qubits):
                raise ValueError("para:{selected_qubits} must be a list of integers.")
            if selected_layers is None:
                selected_layers = [5, 10, 20, 50]
            elif not isinstance(selected_layers, list) or not all(isinstance(_, int) for _ in selected_layers):
                raise ValueError("para:{selected_layers} must be a list of integers.")
            if random_rotation_gate is None:
                random_rotation_gate = ['x', 'y', 'z']
            elif not isinstance(random_rotation_gate, list) or not all(
                    isinstance(gate, str) for gate in random_rotation_gate):
                raise ValueError("para:{random_rotation_gate} must be ['x', 'y', 'z'].")
            if not isinstance(samples, int) or samples <= 0:
                raise ValueError("para:{samples} must be a positive integer.")
            if not isinstance(line_width, int) or line_width <= 0:
                raise ValueError("para:{line_width} must be a positive integer.")
            if not isinstance(bar_width, float) or bar_width <= 0:
                raise ValueError("para:{bar_width} must be a positive float.")
            if not isinstance(font_size, int) or font_size <= 0:
                raise ValueError("para:{font_size} must be a positive integer.")
            if not isinstance(legend_size, int) or legend_size <= 0:
                raise ValueError("para:{legend_size} must be a positive integer.")
            if not isinstance(label_size, int) or label_size <= 0:
                raise ValueError("para:{label_size} must be a positive integer.")
            if not isinstance(absolute, bool):
                raise ValueError("para:{absolute} must be a bool.")
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

        plt.style.use(['science', 'no-latex'])
        self.selected_qubits = selected_qubits
        self.selected_layers = selected_layers
        self.line_width = line_width
        self.bar_width = bar_width
        self.font_size = font_size
        self.legend_size = legend_size
        self.label_size = label_size
        self.absolute = absolute

    @staticmethod
    def relist(data: list[dict], refer_key: str):
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
                raise ValueError('para{refer_key} must be qubit or layer.')
        except ValueError as e:
            print('Error parameter:', e)
            raise
        values = []
        for i in data:
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
        self.original_data = []
        self.modified_data = []
        db = sqlite3.connect('barren/data.db')
        cursor = db.cursor()
        cursor.execute("SELECT * FROM single")
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        for r in rows:
            row = dict(zip(columns, r))
            row["paras"] = json.loads(row["paras"])
            row['gradients'] = json.loads(row['gradients'])
            if bool(row['modified']):
                self.modified_data.append(row)
            else:
                self.original_data.append(row)
        self.qubits = self.relist(self.original_data, 'qubit')
        self.layers = self.relist(self.original_data, 'layer')
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
        This function is plot the relationship between qubits and outputs.
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
                raise ValueError('para:{scatter} must be a bool')
            if not isinstance(bar, bool):
                raise ValueError('para:{bar} must be a bool')
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
        This function is plot the relationship between qubits and gradients.
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
                raise ValueError('para:{scatter} must be a bool')
            if not isinstance(bar, bool):
                raise ValueError('para:{bar} must be a bool')
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
        This function is plot the relationship between qubits and the variance of the gradient.
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
                raise ValueError('para:{refer_layer} must be a positive integer')
            if refer_layer not in self.layers:
                raise ValueError('para:{refer_layer} can not be found')
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
        plt.semilogy(qubits, original_variance, "o", label='Unitary 2-design')
        plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Unitary 2-design:Slope {:3.2f}".format(p[0]), linewidth=self.line_width)
        plt.semilogy(qubits, modified_variance, 'o', label="Unitary 1-design")
        plt.semilogy(qubits, np.exp(q[0] * qubits + q[1]), "o-.", label="Unitary 1-design:Slope {:3.2f}".format(q[0]), linewidth=self.line_width)
        plt.xlabel(r"N Qubits", fontsize=self.font_size)
        plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance", fontsize=self.font_size, fontweight='bold')
        plt.legend(fontsize=self.legend_size)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize=self.label_size, width=3)
        plt.savefig('Experiment_1_1.png', dpi=300)
        # plt.show()

    def layers_variance(self):
        """
                This function is plot the relationship between layers and the variance of the gradient.
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
                plt.plot(self.layers, original_data[index], marker='*', label='Unitary 2-design', linewidth=self.line_width, color='green')
            else:
                plt.plot(self.layers, original_data[index], marker='*', linewidth=self.line_width, color='green', alpha=0.91-0.07*index)
        for index, qubit in enumerate(self.qubits):
            if index == 0:
                plt.plot(self.layers, modified_data[index], marker='o', label='Unitary 1-design', linewidth=self.line_width, color='red')
            else:
                plt.plot(self.layers, modified_data[index], marker='o', linewidth=self.line_width, color='red', alpha=0.91-0.07*index)
        plt.xlabel(r"Layers", fontsize=self.font_size, fontweight='bold')
        plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance", fontsize=self.font_size, fontweight='bold')
        plt.legend(fontsize=self.legend_size)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize=self.label_size, width=3)
        plt.savefig('Experiment_1_2.png', dpi=300)
        # plt.show()

    def exp_2(self):
        """
                This function is plot the relationship between layers and the variance of the gradient.
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

        proposed_data = [[0.1375666437471599, 0.12491017998027969, 0.07846881611077955, 0.10436275392399227, 0.12844288371280158, 0.1388467156850008, 0.10371237140728017, 0.10329554935417502, 0.07526649298145524, 0.09420846329779825, 0.1357111814531118], [0.05251696184014928, 0.03533705699267718, 0.01283533140208281, 0.02143375614698152, 0.022865559119163815, 0.025714138742381515, 0.020291295873931414, 0.020759497465755637, 0.020539232499200803, 0.014060551319738377, 0.017827708316857226], [0.06071482429936492, 0.03984350052718191, 0.0170240528361168, 0.024854795398121892, 0.02210435646083292, 0.021430733620591162, 0.025161176559996556, 0.02215264799668699, 0.021744391610166468, 0.017553293863324923, 0.022100296230311487], [0.06393095823949951, 0.02992482221449466, 0.018542163253285598, 0.02845760665935402, 0.03114551471750895, 0.01976555699223226, 0.02028288102043567, 0.03266166552331013, 0.02476044616313271, 0.020075655640874754, 0.019015448402947218], [0.05771769398793716, 0.04087971729484063, 0.020380556317189543, 0.028293835370193438, 0.025367016166849922, 0.028871989716100634, 0.03178595175220831, 0.03079046996028728, 0.024108418790407846, 0.02496836546606571, 0.025275220345397566], [0.0480122971154398, 0.05005190084991863, 0.013788519266435635, 0.016812686614134893, 0.019540601531597205, 0.021196624530961154, 0.013513901162529925, 0.018242963871579972, 0.02045253906519503, 0.021111340341797985, 0.022031988862406576], [0.05884753158454884, 0.039190698125738714, 0.01837630158710517, 0.013605676828287604, 0.022229050745151472, 0.015141775329452756, 0.01579034412300152, 0.01714773980733852, 0.012477903851424832, 0.024137990481216044, 0.01891118669956203], [0.04380858214829597, 0.03639710074659648, 0.02357036267383989, 0.018506977761391204, 0.018442581407107653, 0.024773059371142655, 0.016403622069126052, 0.024163661639043398, 0.016364967608442298, 0.029390969849569126, 0.01565264380531861], [0.05165953759982704, 0.03867103207430761, 0.019419706143717966, 0.02339236670301256, 0.01532512177702378, 0.018967101439380255, 0.020101399261149747, 0.024064884288963945, 0.02471461572369276, 0.01953280024003751, 0.016920947277530102]]

        original_data = [[0.0921305130601163, 0.1201095864248526, 0.10462086550506777, 0.07782599968534858, 0.07092286599119652, 0.09038234773252395, 0.06189990018468901, 0.07943113762726718, 0.10313034541435234, 0.09894776374428049, 0.08395008463629942],
         [0.07150929845574121, 0.03249049700665185, 0.01759060071405434, 0.019530007986277576, 0.02488166172503422, 0.02307562430900119, 0.026258670184189987, 0.022390288958042178, 0.021859254569683628, 0.019937726547538517, 0.016549632449640164],
         [0.05267133463986978, 0.03808526033060597, 0.01850806435390122, 0.005120912390921583, 0.005659713938448837, 0.005518203052428133, 0.00586881227705096, 0.004274985821350007, 0.004076611320528497, 0.006164786645825964, 0.003433085037522245],
         [0.053357797246422374, 0.03509168233101714, 0.02369399937834707, 0.0025586744798863654, 0.0010908260219356985, 0.001088018609598879, 0.0014505394178455147, 0.001774387360818543, 0.0011913128889283508, 0.0011930995221935184, 0.0009220324266548008],
         [0.06432002569530929, 0.038267583895063226, 0.023571430981000327, 0.0019312506818683626, 0.0004475972909535001, 0.0003164389218343304, 0.0002816716925954457, 0.00027550417039702747, 0.00028762422493436307, 0.00034212172847429735, 0.0002738463231626068],
         [0.053266024201347024, 0.04555892545584994, 0.01768799999971872, 0.002908352971716387, 0.00014499783430791082, 9.925014653705743e-05, 7.717265788882421e-05, 8.756729957823585e-05, 7.526326683355459e-05, 0.00010214317187988352, 8.612432503727374e-05],
         [0.05600855665471349, 0.05455981354783777, 0.017327845920192254, 0.0023346610387647766, 0.0001366644131133145, 2.6985896154716993e-05, 2.2137134213514568e-05, 1.849141622244648e-05, 2.186063810365585e-05, 2.826893321887267e-05, 1.565187610517698e-05],
         [0.05482981156746218, 0.04637441257979377, 0.021271391709068534, 0.001431050038628469, 6.91629757608511e-05, 7.235465261001103e-06, 5.111477069047928e-06, 4.1746964728911125e-06, 4.8551479399861325e-06, 4.301962864671711e-06, 4.667365835277301e-06],
         [0.0606673705008715, 0.033079111355495974, 0.017994114329758573, 0.002071393806930489, 8.5116137177719e-05, 7.431909439212301e-06, 1.8918546829881035e-06, 1.2644654868030701e-06, 1.4318234022131306e-06, 1.5815240554847327e-06, 1.185652983966273e-06]]

        modified_data = [[0.012956754096441947, 0.009254436554730875, 0.011809345762642814, 0.012035095470163017, 0.01288463405136459, 0.014500470816247337, 0.01226726242103122, 0.010640468571908899, 0.008829699968869701, 0.01466923670226916, 0.010426533408447892],
         [0.009114514059847449, 0.009795037381411414, 0.008854300235594253, 0.008420958034989733, 0.006551950569510405, 0.008415796285745769, 0.006896400168636174, 0.006526883861986869, 0.009482150428030505, 0.0073648237529185655, 0.005870460839684479],
         [0.009483885316358905, 0.009470718422597744, 0.00865598791302703, 0.008306222121748504, 0.00628010312508339, 0.008951126693580865, 0.007114191123105031, 0.007230386198815393, 0.008094559194177174, 0.010113350285563972, 0.0056252212647749],
         [0.008355442120857199, 0.009432283371566739, 0.008775487431363865, 0.0046704861319354285, 0.009615730021411617, 0.005442804109855986, 0.008022570570555596, 0.007993965277863476, 0.009031168689989287, 0.005400931845998298, 0.008353873989751574],
         [0.004606964965166829, 0.010417157801239465, 0.008087315636347769, 0.00906241032386823, 0.012828450658221649, 0.0060232315508780086, 0.00915499258165792, 0.008843870145184014, 0.009126447611699957, 0.007043957943433988, 0.007931813101946236],
         [0.009833966637548928, 0.00970333009360362, 0.009339249876669573, 0.005873849887784799, 0.008734212436222074, 0.005942419066332642, 0.008305317922104934, 0.0060785066677510715, 0.0061652604561474545, 0.0066103142835516335, 0.00671281779790487],
         [0.009975647394179065, 0.011553911861340842, 0.0079553753069193, 0.007460788239981674, 0.007441194107323881, 0.009777207175697107, 0.006325100086584013, 0.006463004015732296, 0.0085471213120022, 0.007867817305507589, 0.007374107347340474],
         [0.004891815827273291, 0.011090780707053058, 0.009691296100508767, 0.009049248273368754, 0.00520409342436794, 0.007394645676011568, 0.008973445528185521, 0.006702262554426281, 0.007950033261733484, 0.007730431802872767, 0.006676561024237382],
         [0.006752574743444711, 0.01275109725592813, 0.008723154781364907, 0.0075977126837717315, 0.008477235764856556, 0.00743661075496124, 0.005370379659871337, 0.007971226628947716, 0.0072553194529525545, 0.007981173512283501, 0.00782023180529803]]

        self.qubits = [2, 4, 6, 8, 10, 12, 14, 16]

        # Plot the line for each qubit
        plt.figure(figsize=(16, 9))
        for index, qubit in enumerate(self.qubits):
            if index == 0:
                plt.plot(self.layers, original_data[index], marker='*', label='Unitary 2-design', linewidth=self.line_width, color='green')
            else:
                plt.plot(self.layers, original_data[index], marker='*', linewidth=self.line_width, color='green', alpha=0.91-0.07*index)
        for index, qubit in enumerate(self.qubits):
            if index == 0:
                plt.plot(self.layers, modified_data[index], marker='o', label='Unitary 1-design', linewidth=self.line_width, color='red')
            else:
                plt.plot(self.layers, modified_data[index], marker='o', linewidth=self.line_width, color='red', alpha=0.91-0.07*index)
        for index, qubit in enumerate(self.qubits):
            if index == 0:
                plt.plot(self.layers, proposed_data[index], marker='x', label='Proposed Structure', linewidth=self.line_width, color='purple')
            else:
                plt.plot(self.layers, proposed_data[index], marker='x', linewidth=self.line_width, color='purple', alpha=0.91-0.07*index)
        plt.xlabel(r"Layers", fontsize=self.font_size, fontweight='bold')
        plt.ylabel(r"Variance of $\partial \theta_{1, 1} E$", fontsize=self.font_size, fontweight='bold')
        plt.legend(fontsize=self.legend_size)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize=self.label_size, width=3)
        plt.savefig('Experiment_2_2.png', dpi=300)
        # plt.show()

    def ex1(self, refer_layer: int = 500):
        """
        This function is plot the relationship between qubits and the variance of the gradient.
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
                raise ValueError('para:{refer_layer} must be a positive integer')
            if refer_layer not in self.layers:
                raise ValueError('para:{refer_layer} can not be found')
        except ValueError as e:
            print('Error parameter:', e)
            raise

        self.qubits = [2, 4, 6, 8, 10, 12, 14, 16]
        proposed_data = [[0.1375666437471599, 0.12491017998027969, 0.07846881611077955, 0.10436275392399227, 0.12844288371280158, 0.1388467156850008, 0.10371237140728017, 0.10329554935417502, 0.07526649298145524, 0.09420846329779825, 0.1357111814531118], [0.05251696184014928, 0.03533705699267718, 0.01283533140208281, 0.02143375614698152, 0.022865559119163815, 0.025714138742381515, 0.020291295873931414, 0.020759497465755637, 0.020539232499200803, 0.014060551319738377, 0.017827708316857226], [0.06071482429936492, 0.03984350052718191, 0.0170240528361168, 0.024854795398121892, 0.02210435646083292, 0.021430733620591162, 0.025161176559996556, 0.02215264799668699, 0.021744391610166468, 0.017553293863324923, 0.022100296230311487], [0.06393095823949951, 0.02992482221449466, 0.018542163253285598, 0.02845760665935402, 0.03114551471750895, 0.01976555699223226, 0.02028288102043567, 0.03266166552331013, 0.02476044616313271, 0.020075655640874754, 0.019015448402947218], [0.05771769398793716, 0.04087971729484063, 0.020380556317189543, 0.028293835370193438, 0.025367016166849922, 0.028871989716100634, 0.03178595175220831, 0.03079046996028728, 0.024108418790407846, 0.02496836546606571, 0.025275220345397566], [0.0480122971154398, 0.05005190084991863, 0.013788519266435635, 0.016812686614134893, 0.019540601531597205, 0.021196624530961154, 0.013513901162529925, 0.018242963871579972, 0.02045253906519503, 0.021111340341797985, 0.022031988862406576], [0.05884753158454884, 0.039190698125738714, 0.01837630158710517, 0.013605676828287604, 0.022229050745151472, 0.015141775329452756, 0.01579034412300152, 0.01714773980733852, 0.012477903851424832, 0.024137990481216044, 0.01891118669956203], [0.04380858214829597, 0.03639710074659648, 0.02357036267383989, 0.018506977761391204, 0.018442581407107653, 0.024773059371142655, 0.016403622069126052, 0.024163661639043398, 0.016364967608442298, 0.029390969849569126, 0.01565264380531861], [0.05165953759982704, 0.03867103207430761, 0.019419706143717966, 0.02339236670301256, 0.01532512177702378, 0.018967101439380255, 0.020101399261149747, 0.024064884288963945, 0.02471461572369276, 0.01953280024003751, 0.016920947277530102]]

        index = self.layers.index(refer_layer)
        proposed_variance = [proposed_data[i][index] for i in range(len(self.qubits))]

        original_variance = [0.08395008463629942, 0.016549632449640164, 0.003433085037522245, 0.0009220324266548008, 0.0002738463231626068, 8.612432503727374e-05, 1.565187610517698e-05, 4.667365835277301e-06]
        modified_variance = [0.010426533408447892, 0.005870460839684479, 0.0056252212647749, 0.008353873989751574, 0.007931813101946236, 0.00671281779790487, 0.007374107347340474, 0.006676561024237382]

        p = np.polyfit(self.qubits, np.log(original_variance), 1)  # original poly fitting
        q = np.polyfit(self.qubits, np.log(modified_variance), 1)  # modified poly fitting
        qubits = np.array(self.qubits)
        # Plot the straight line fit to the semi-logy
        plt.figure(figsize=(16, 9))
        plt.semilogy(qubits, original_variance, "o", label='Unitary 2-design', color='blue')
        plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Unitary 2-design:Slope {:3.2f}".format(p[0]), linewidth=self.line_width, color='green')
        plt.semilogy(qubits, modified_variance, 'o', label="Unitary 1-design", color='orange')
        plt.semilogy(qubits, np.exp(q[0] * qubits + q[1]), "o-.", label="Unitary 1-design:Slope {:3.2f}".format(q[0]), linewidth=self.line_width, color='red')
        plt.semilogy(qubits, proposed_variance, 'o-', label='Proposed Structure', linewidth=self.line_width, color='purple')
        plt.xlabel(r"N Qubits", fontsize=self.font_size)
        plt.ylabel(r"Variance of $\partial \theta_{1, 1} E$", fontsize=self.font_size, fontweight='bold')
        plt.legend(fontsize=self.legend_size)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize=self.label_size, width=3)
        plt.savefig('Experiment_2_1.png', dpi=300)
        # plt.show()