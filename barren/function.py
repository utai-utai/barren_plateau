from .barren_plateau import BP
from .barren_plateau_multi import BPs
from .plotting import PLOTTING, plot_simple_datas


def bp(**kwargs) -> list[dict]:
    modify = kwargs.get('modify', False)
    qubits = kwargs.get('qubits', None)
    layers = kwargs.get('layers', None)
    random_rotation_gate = kwargs.get('random_rotation_gate', None)
    samples = kwargs.get('samples', 100)
    save = kwargs.get('save', False)
    simulated_bp = BP(modify=modify, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate, samples=samples, save=save)
    return simulated_bp.run()


def bps(**kwargs) -> list[dict]:
    modify = kwargs.get('modify', False)
    qubits = kwargs.get('qubits', None)
    layers = kwargs.get('layers', None)
    num_paras = kwargs.get('num_paras', -1)
    random_rotation_gate = kwargs.get('random_rotation_gate', None)
    samples = kwargs.get('samples', 100)
    save = kwargs.get('save', False)
    simulated_bps = BPs(modify=modify, qubits=qubits, layers=layers, num_paras=num_paras, random_rotation_gate=random_rotation_gate, samples=samples, save=save)
    return simulated_bps.run()


def train(**kwargs) -> list[dict]:
    modify = kwargs.get('modify', False)
    qubits = kwargs.get('qubits', None)
    layers = kwargs.get('layers', None)
    num_paras = kwargs.get('num_paras', -1)
    random_rotation_gate = kwargs.get('random_rotation_gate', None)
    simulated_bps = BPs(modify=modify, qubits=qubits, layers=layers, num_paras=num_paras, random_rotation_gate=random_rotation_gate)
    target = kwargs.get('target', 0.1)
    epochs = kwargs.get('epochs', 100)
    lr = kwargs.get('lr', 0.05)
    layer_decrease_rate = kwargs.get('layer_decrease_rate', -0.5)
    return simulated_bps.train(target=target, epochs=epochs, lr=lr, layer_decrease_rate=layer_decrease_rate)


def plot_qubit_gradient(**kwargs):
    saved_data = kwargs.get('saved_data', False)
    if saved_data:
        plotting = PLOTTING(saved_data=saved_data)
    else:
        original_data = kwargs.get('original_data', None)
        modified_data = kwargs.get('modified_data', None)
        selected_qubits = kwargs.get('select_qubits', None)
        selected_layers = kwargs.get('select_layers', None)
        random_rotation_gate = kwargs.get('random_rotation_gate', None)
        samples = kwargs.get('samples', 100)
        line_width = kwargs.get('line_width', 3)
        bar_width = kwargs.get('bar_width', 0.01)
        font_size = kwargs.get('font_size', 30)
        legend_size = kwargs.get('legend_size', 25)
        label_size = kwargs.get('label_size', 30)
        plotting = PLOTTING(original_data=original_data, modified_data=modified_data, selected_qubits=selected_qubits, selected_layers=selected_layers, random_rotation_gate=random_rotation_gate, samples=samples, line_width=line_width, bar_width=bar_width, font_size=font_size, legend_size=legend_size, label_size=label_size)
    scatter = kwargs.get('scatter', True)
    bar = kwargs.get('bar', True)
    plotting.qubit_gradient(scatter=scatter, bar=bar)


def plot_qubits_variance(**kwargs):
    saved_data = kwargs.get('saved_data', False)
    if saved_data:
        plotting = PLOTTING(saved_data=saved_data)
        refer_layer = kwargs.get('refer_layer', 500)
    else:
        original = kwargs.get('original_data', None)
        modified = kwargs.get('modified_data', None)
        qubits = []
        layers = []
        for i in original:
            qubits.append(i['qubit'])
            layers.append(i['layer'])
        qubits = sorted(set(qubits))
        layers = sorted(set(layers))
        original_data = []
        modified_data = []
        for qubit in qubits:
            original_temp = []
            modified_temp = []
            for layer in layers:
                original_temp.extend(i['variance'] for i in original if i.get('qubit') == qubit and i.get('layer') == layer)
                modified_temp.extend(i['variance'] for i in modified if i.get('qubit') == qubit and i.get('layer') == layer)
            original_data.append(original_temp)
            modified_data.append(modified_temp)
        random_rotation_gate = kwargs.get('random_rotation_gate', None)
        samples = kwargs.get('samples', 100)
        line_width = kwargs.get('line_width', 3)
        bar_width = kwargs.get('bar_width', 0.01)
        font_size = kwargs.get('font_size', 30)
        legend_size = kwargs.get('legend_size', 25)
        label_size = kwargs.get('label_size', 30)
        refer_layer = kwargs.get('refer_layer', layers[-1])
        plotting = PLOTTING(original_data=original_data, modified_data=modified_data, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate, samples=samples, line_width=line_width, bar_width=bar_width, font_size=font_size, legend_size=legend_size, label_size=label_size)
    plotting.qubits_variance(refer_layer=refer_layer)


def plot_layers_variance(**kwargs):
    saved_data = kwargs.get('saved_data', False)
    if saved_data:
        plotting = PLOTTING(saved_data=saved_data)
    else:
        original = kwargs.get('original_data', None)
        modified = kwargs.get('modified_data', None)
        qubits = []
        layers = []
        for i in original:
            qubits.append(i['qubit'])
            layers.append(i['layer'])
        qubits = sorted(set(qubits))
        layers = sorted(set(layers))
        original_data = []
        modified_data = []
        for qubit in qubits:
            original_temp = []
            modified_temp = []
            for layer in layers:
                original_temp.extend(i['variance'] for i in original if i.get('qubit') == qubit and i.get('layer') == layer)
                modified_temp.extend(i['variance'] for i in modified if i.get('qubit') == qubit and i.get('layer') == layer)
            original_data.append(original_temp)
            modified_data.append(modified_temp)
        random_rotation_gate = kwargs.get('random_rotation_gate', None)
        samples = kwargs.get('samples', 100)
        line_width = kwargs.get('line_width', 3)
        bar_width = kwargs.get('bar_width', 0.01)
        font_size = kwargs.get('font_size', 30)
        legend_size = kwargs.get('legend_size', 25)
        label_size = kwargs.get('label_size', 30)
        plotting = PLOTTING(original_data=original_data, modified_data=modified_data, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate, samples=samples, line_width=line_width, bar_width=bar_width, font_size=font_size, legend_size=legend_size, label_size=label_size)
    plotting.layers_variance()


def plot_results(**kwargs):
    name = kwargs.get('name', None)
    original_data = kwargs.get('original_data', None)
    original = original_data[-1][name]
    modified_data = kwargs.get('modified_data', None)
    modified = modified_data[-1][name]
    scatter = kwargs.get('scatter', True)
    bar = kwargs.get('bar', True)
    plot_simple_datas(original=original, modified=modified, name=name, scatter=scatter, bar=bar)
