import random as rand
from timeit import default_timer as timer
import multiprocessing as mp
import torch
import torch.nn as nn
import joblib
import subprocess
import hashlib

n = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(A):
    model = nn.Sequential(
        nn.Linear(in_features=36, out_features=64),
        nn.Tanh(),
        # nn.Linear(in_features=64, out_features=64),
        # nn.Tanh(),
        # nn.Linear(in_features=64, out_features=64),
        # nn.Tanh(),
        # nn.Linear(in_features=64, out_features=64),
        # nn.Tanh(),
        nn.Linear(in_features=64, out_features=64),
        nn.Tanh(),
        nn.Linear(in_features=64, out_features=1),
    ).to(device)
    model.load_state_dict(torch.load('fixed_ideal_time_guesser.pth',
                                     map_location=device))
    model.eval()
    scaler_x = joblib.load('scaler_x.pkl')
    A = torch.tensor(
        scaler_x.transform([A]),
        dtype=torch.float32).to(device)
    with torch.inference_mode():
        output = model(A)
    return float(output[0][0])

def main():
    s = timer()
    totaltime = 0
    TIME_LIMIT = 600
    NUM_SAMPLES = 200
    MATRIX_ENTRY_SIZE = 100

    mintime = 1000
    for _ in range(15):
        A = [rand.randint(1, MATRIX_ENTRY_SIZE) for _ in range(n*n)]
        pred = predict(A)
        print(f"trying the matrix {A}, expecting {predict(A)}")
        result = subprocess.check_output(['./gb_timer.sage'] + [str(i) for i in A]\
            + [str(TIME_LIMIT)])
        print(f"actual time: {result.decode("utf-8")}")

if __name__ == "__main__":
    main()
