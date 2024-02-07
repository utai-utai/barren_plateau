from plotting import PLOTTING

if __name__ == '__main__':
    layers = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500]
    f = PLOTTING(saved_data=True)
    f.qubit_gradient()
