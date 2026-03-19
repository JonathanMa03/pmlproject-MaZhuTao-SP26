import pandas as pd
import numpy as np

from models.gaussian_bvar import GaussianBVAR
from models.student_t_bvar import StudentTBVAR
from models.mixture_bvar import MixtureBVAR
from models.sv_bvar import SVBVAR


def load_data():
    df = pd.read_csv("data.csv")
    return df.values


def main():

    Y = load_data()

    print("Running Gaussian VAR")
    g = GaussianBVAR(p=2)
    g.fit(Y)

    print("Running Student-t VAR")
    t = StudentTBVAR(p=2)
    t.fit(Y)

    print("Running Mixture VAR")
    m = MixtureBVAR(p=2)
    m.fit(Y)

    print("Running SV VAR")
    sv = SVBVAR(p=2)
    sv.fit(Y)

    print("Done")


if __name__ == "__main__":
    main()