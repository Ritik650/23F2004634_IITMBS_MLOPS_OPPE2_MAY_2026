"""
common/preprocessing.py
------------------------
`encode_gender` is wrapped in a sklearn FunctionTransformer inside the
training Pipeline (see training/train.py). joblib pickles a reference
to this function's *module path*, not its code — so at inference time,
whatever process unpickles the model (app/main.py, running inside the
serving container) must be able to `import common.preprocessing` and
find a function with this exact name.

This is why the function lives in its own small shared module, copied
into BOTH Docker build stages (see Dockerfile), rather than being
defined inline in training/train.py: if it only existed in the
training stage, the serving container's unpickler would fail with
`AttributeError: Can't get attribute 'encode_gender'` — which is
exactly what happened before this file existed.
"""

import pandas as pd


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map({"male": 1, "female": 0}).astype(float)
    return df
