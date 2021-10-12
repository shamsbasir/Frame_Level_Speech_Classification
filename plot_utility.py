from   configs import newDir
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime

def save_output(vector,name):

    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print(" Saving outputs ...")
    newDir('output')
    PATH  = os.getcwd()+"/output/"
    np.save(PATH+ f'{name}_{current_time}.npy',vector)


def visualize(soln_vect,name):
    print(f"Saving graphs for \"{name}\"")
    newDir('output')
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    PATH = os.getcwd() + "/output/"
    try :
        plt.plot(soln_vect)
        plt.ylabel(name)
        plt.savefig(PATH +f"{name}_{current_time}.png")
    except Exception as e:
        traceback.print_exc()
        print("Error: Problems generating plots. See if a .png was generated in output folder\n")


def kagglize_output_labels(vector,name):
    pass

if __name__=="__main__":

    a = np.random.rand(10)
    b = a+3
    c = b+3

    save_output(a,'a')
    save_output (b,'b')
    save_output (c,'c')

    visualize(a,'a')
    visualize(b,'b')
    visualize(c,'c')
