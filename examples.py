from barren import *

if __name__ == '__main__':
    '''example 1: barren plateau of performance on parameter:θ_{1, 1}'''
    original_detail = bp(qubits=[2, 4, 6], layers=[5, 10, 20])
    modified_detail = bp(qubits=[2, 4, 6], layers=[5, 10, 20], modify=True)
    plot_qubit_gradient(original_data=original_detail, modified_data=modified_detail)
    plot_qubits_variance(original_data=original_detail, modified_data=modified_detail)
    plot_layers_variance(original_data=original_detail, modified_data=modified_detail)
    # use saved data
    plot_qubit_gradient(saved_data=True)
    plot_qubits_variance(saved_data=True)
    plot_layers_variance(saved_data=True)

    '''example 2: barren plateau of performance on all parameters'''
    original_details = bps(qubits=[2, 4], layers=[5, 10])
    modified_details = bps(qubits=[2, 4], layers=[5, 10], modify=True)
    plot_qubits_variance(original_data=original_details, modified_data=modified_details)
    plot_layers_variance(original_data=original_details, modified_data=modified_details)

    '''example 3: train a simple network'''
    original_train_detail = train(qubits=[8], layers=[50], target=0.1)
    modified_train_detail = train(qubits=[8], layers=[50], target=0.1, modify=True)
    print(original_train_detail, modified_train_detail)
    plot_results(original_data=original_train_detail, modified_data=modified_train_detail, name='output')
    plot_results(original_data=original_train_detail, modified_data=modified_train_detail, name='cost')

    '''example 4: experiments, you can use (save=True) to save the results.'''
    ex1()
    ex2()
    ex3(save=True)
