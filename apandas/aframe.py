from typing import Iterable
import pandas as pd
from .acolumn import AColumn


class AFrame(pd.DataFrame):
    def __init__(self, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def __getitem__(self, key):
        if isinstance(key, AColumn):
            if not key.name in self.columns:
                self.add_acolumn(key)
            return super().__getitem__(key)
        elif not isinstance(key, str) and isinstance(key, Iterable):
            orig_key = key
            key = []
            for k in orig_key:
                if isinstance(k, AColumn):
                    if not k.name in self.columns:
                        self.add_acolumn(k)
                    key.append(str(k))
                else:
                    key.append(k)
            return AFrame(super().__getitem__(key))
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
                print(f'Key "{acol}" not found in the AFrame, adding.')
            self[acol] = acol.func.func(self)

    def _fix_columns_kwarg(self, kwargs):
        if 'columns' in kwargs:
            columns = kwargs['columns']
            if isinstance(columns, dict):
                str_columns = {}
                for col, new_col in columns.items():
                    str_new_col = new_col.name if isinstance(new_col, AColumn) else new_col
                    if isinstance(col, AColumn):
                        self.add_acolumn(col)
                        str_columns[str(col)] = str_new_col
                    else:
                        str_columns[col] = str_new_col
                kwargs['columns'] = str_columns
            else:
                if isinstance(columns, str) or not(isinstance(columns, Iterable)
                                                   or isinstance(columns, AColumn)):
                    str_columns = columns
                elif isinstance(columns, AColumn):
                    self.add_acolumn(columns)
                    str_columns = str(columns)
                else:
                    str_columns = []
                    for col in columns:
                        if isinstance(col, AColumn):
                            self.add_acolumn(col)
                            str_columns.append(str(col))
                        else:
                            str_columns.append(col)
                kwargs['columns'] = str_columns
        return kwargs

    def rename(self, *args, **kwargs):
        """ AFrame supports renaming of AColumns if specified via `columns` kwarg."""
        return AFrame(super().rename(*args, **self._fix_columns_kwarg(kwargs)))

    def drop(self, *args, **kwargs):
        """ AFrame supports dropping of AColumns if specified via `columns` kwarg."""
        return AFrame(super().drop(*args, **self._fix_columns_kwarg(kwargs)))

    def copy(self, *args, **kwargs):
        return AFrame(super().copy(*args, *kwargs))

