# Barren Plateau

Simulate the phenomenon of the barren plateau.

## Installation

```bash
$ pip install -r barren/requirements.txt
```

## Abstract

In the search for quantum advantage with near-term quantum devices, navigating the optimization landscape is significantly hampered by the barren plateaus phenomenon. 
This study presents a strategy to overcome this obstacle without changing the quantum circuit architecture. 
We propose incorporating auxiliary control qubits to shift the circuit from a unitary $2$-design to a unitary $1$-design, mitigating the prevalence of barren plateaus. We then remove these auxiliary qubits to return to the original circuit structure while preserving the unitary $1$-design properties.
Our experiment suggests that the proposed structure effectively mitigates the barren plateaus phenomenon. 
A significant experimental finding is that the gradient of $\theta_{1,1}$, the first parameter in the quantum circuit, displays a broader distribution as the number of qubits and layers increases.
This suggests a higher probability of obtaining effective gradients.
This stability is critical for the efficient training of quantum circuits, especially for larger and more complex systems. The results of this study represent a significant advance in the optimization of quantum circuits and offer a promising avenue for the scalable and practical implementation of quantum computing technologies. This approach opens up new opportunities in quantum learning and other applications that require robust quantum computing power.

## Citation
https://arxiv.org/abs/2406.03748
