from barren_plateau import BP
from plotting import PLOTTING

if __name__ == '__main__':
    # prepare parameters
    # 1. large data
    # qubits = [2*i for i in range(1, 10)]
    # layers = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500]
    # 2. small data (default)

    # print('example 1: print the variance')
    # bp = BP()
    # results = bp.get_results()
    # print(results)
    #
    # print('example 2: save data')
    # bp = BP(save=True)

    f = PLOTTING(saved_data=True)
    print('example 3: plot qubit-outputs')
    f.qubit_output()

    print('example 4: plot qubit-gradients')
    f.qubit_gradient()

    print('example 5: plot qubits-variance')
    f.qubits_variance(refer_layer=50)

    print('example 6: plot layers-variance')
    f.layers_variance()