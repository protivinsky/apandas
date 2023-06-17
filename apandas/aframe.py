from typing import Iterable
import pandas as pd
from apandas import AColumn


class AFrame(pd.DataFrame):
    def __init__(self, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def __getitem__(self, key):
        if isinstance(key, AColumn):
            if not key.name in self.columns:
                self.add_acolumn(key)
            key = str(key)
        elif isinstance(key, Iterable):
            orig_key = key
            key = []
            for k in orig_key:
                if isinstance(k, AColumn):
                    if not k.name in self.columns:
                        self.add_acolumn(k)
                    key.append(str(k))
                else:
                    key.append(k)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, AColumn):
            key = str(key)
        elif isinstance(key, Iterable):
            key = [str(k) if isinstance(key, AColumn) else k for k in key]
        return super().__setitem__(key, value)

    def add_acolumn(self, acol: AColumn):
        """ Generate `acol AColumn` in the `AFrame`. """
        if not acol.name in self.columns:
            if self.verbose:
                print(f'Key "{acol}" not found in the Frame, adding.')
            self[acol] = acol.func.func(self)

