import pandas as pd
from dvc.repo import Repo

def push_file_to_dvc(data: dict, data_name: str):

    df = pd.DataFrame(data)
    repo = Repo()

    data_filename = f"data/{data_name}.csv"
    df.to_csv(data_filename, index=False)

    repo.add(data_filename)
    repo.push()