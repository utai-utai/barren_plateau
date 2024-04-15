import numpy as np
from matplotlib import pyplot as plt
import scienceplots

layers = [5, 10, 20, 50, 100, 150, 200, 300]
line_width: int = 3
font_size: int = 30
legend_size: int = 15
label_size: int = 30


def ex1(save: bool = False):
    original_variance = [0.10554527557157171, 0.053045272904359723, 0.02816030059076791, 0.014221600303922013, 0.007383081470391254, 0.0032930440906034884, 0.0018395284162567834, 0.0013688413786579893, 0.001018590799341449, 0.0005905207245605735, 0.0002578524054853119, 0.00014849875937670166, 8.23561664559645e-05, 5.3296825533957924e-05, 2.5138393498086368e-05]
    modified_variance = [0.013955921457601423, 0.004326479729005139, 0.002469609591456336, 0.0017415632819577508, 0.0016579538958708634, 0.0013445240900041803, 0.0011795560881166548, 0.001043079809910853, 0.0009223940267061373, 0.0005749513633400306, 0.00038313897454460944, 0.00024715409343765765, 0.0001756855720028007, 0.0001548220943292028, 0.00011470349673090366]

    qubits = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    p = np.polyfit(qubits, np.log(original_variance), 1)  # original poly fitting
    q = np.polyfit(qubits, np.log(modified_variance), 1)  # modified poly fitting
    qubits = np.array(qubits)

    # Plot the straight line fit to the semi-logy
    plt.figure(figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])
    plt.semilogy(qubits, original_variance, "o", label='Unitary 2-design')
    plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Unitary 2-design:Slope {:3.2f}".format(p[0]), linewidth=line_width)
    plt.semilogy(qubits, modified_variance, 'o', label="Unitary 1-design")
    plt.semilogy(qubits, np.exp(q[0] * qubits + q[1]), "o-.", label="Unitary 1-design:Slope {:3.2f}".format(q[0]), linewidth=line_width)
    plt.xlabel(r"N Qubits", fontsize=font_size)
    plt.ylabel(r"Variance of $\partial_k E$", fontsize=font_size, fontweight='bold')
    plt.legend(fontsize=legend_size)
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=label_size, width=3)
    if save:
        plt.savefig('Experiment_1.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    ex1()