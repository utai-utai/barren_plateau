from .barren_plateau import BP
from .barren_plateau_multi import BPs


def bp(**kwargs) -> list[dict]:
    modify = kwargs.get('modify', False)
    qubits = kwargs.get('qubits', None)
    layers = kwargs.get('layers', None)
    random_rotation_gate = kwargs.get('random_rotation_gate', None)
    samples = kwargs.get('samples', 100)
    save = kwargs.get('save', False)
    simulated_bp = BP(modify=modify, qubits=qubits, layers=layers, random_rotation_gate=random_rotation_gate, samples=samples, save=save)
    return simulated_bp.run()


def bps_gradient(**kwargs) -> list[dict]:
    modify = kwargs.get('modify', False)
    qubits = kwargs.get('qubits', None)
    layers = kwargs.get('layers', None)
    num_paras = kwargs.get('num_paras', -1)
    random_rotation_gate = kwargs.get('random_rotation_gate', None)
    samples = kwargs.get('samples', 100)
    save = kwargs.get('save', False)
    simulated_bps = BPs(modify=modify, qubits=qubits, layers=layers, num_paras=num_paras, random_rotation_gate=random_rotation_gate, samples=samples, save=save)
    return simulated_bps.run()


def bps_output(**kwargs) -> list[dict]:
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
