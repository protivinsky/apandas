import functools
from typing import Iterable
import pandas as pd
import tree
from .acolumn import AColumn


class AMeta(type):
    """
    A metaclass for AFrame (and AFrameGroupBy) that modifies all the methods of the parent class
    so they are compatible with using AColumn arguments instead of strings as column names.

    Admittedly this is a bit hacky, but it does what I need.
    """

    def __new__(mcs, name, bases, attrs):
        aclass = super().__new__(mcs, name, bases, attrs)
        parent_class = bases[0]

        def any_acol_in_tree(t):
            flags = []
            def acol_in_tree(t):
                b = isinstance(t, AColumn) or (isinstance(t, dict) and any(isinstance(k, AColumn) for k in t.keys()))
                flags.append(b)
            tree.traverse(acol_in_tree, t)
            return any(flags)

        def method_wrapper(method):
            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                orig_args = args.__copy__() if hasattr(args, '__copy__') else args
                orig_kwargs = kwargs.copy()

                def map_keys(mapping):
                    if isinstance(mapping, dict):
                        keys = list(mapping.keys())
                        for key in keys:
                            if isinstance(key, AColumn):
                                if method.__name__ != '__setitem__':
                                    if name == 'AFrame':
                                        self.add_acolumn(key)
                                    elif name == 'AFrameGroupBy' and not key.name in self.obj.columns:
                                        self.obj.add_acolumn(key)
                                mapping[key.name] = mapping.pop(key)

                def map_leaves(x):
                    if isinstance(x, AColumn):
                        if method.__name__ != '__setitem__':
                            if name == 'AFrame':
                                self.add_acolumn(x)
                            elif name == 'AFrameGroupBy' and not x.name in self.obj.columns:
                                self.obj.add_acolumn(x)
                        return x.name
                    else:
                        return x

                any_acol_args = any_acol_in_tree(orig_args)
                any_acol_kwargs = any_acol_in_tree(orig_kwargs)
                if any_acol_args:
                    args = tree.traverse(map_keys, args)
                    args = tree.map_structure(map_leaves, args)
                    # print(f'Converting ARGS: {orig_args} --> {args}')
                if any_acol_kwargs:
                    kwargs = tree.traverse(map_keys, kwargs, top_down=False)
                    kwargs = tree.map_structure(map_leaves, kwargs)
                    # print(f'Converting KWARGS: {orig_kwargs} --> {kwargs}')

                # print(f'Calling {method.__name__} with args={args}, kwargs={kwargs}')
                result = method(self, *args, **kwargs)
                if isinstance(result, pd.DataFrame) and not isinstance(result, AFrame):
                    # print(f'Converting pd.DataFrame {result} to an AFrame:')
                    result = AFrame(result)
                if isinstance(result, pd.core.groupby.generic.DataFrameGroupBy) and not isinstance(
                        result, AFrameGroupBy):
                    if isinstance(self, AFrame):
                        result = AFrameGroupBy(self, *args, **kwargs)
                    elif isinstance(self, AFrameGroupBy) and method.__name__ == '__getitem__':
                        result = AFrameGroupBy(self.obj, keys=self.keys, axis=self.axis, as_index=self.as_index,
                                               selection=args[0], group_keys=self.group_keys, dropna=self.dropna)

                return result
            return wrapper

        for attr_name, attr_value in parent_class.__dict__.items():
            if attr_name not in aclass.__dict__ and callable(attr_value):
                # if not attr_name.startswith('_') or (attr_name == '__getitem__' and name == 'AFrameGroupBy'):
                if not attr_name.startswith('_') or attr_name in ['__getitem__', '__setitem__']:
                    print(f'Modifying {attr_name} on {name} from {parent_class.__name__}')
                    setattr(aclass, attr_name, method_wrapper(attr_value))
                else:
                    print(f'Keeping {attr_name} on {name} from {parent_class.__name__}')
                    setattr(aclass, attr_name, attr_value)


        return aclass


class AFrame(pd.DataFrame, metaclass=AMeta):
    """
    AFrame is a subclass of pandas.DataFrame that allows to use AColumn objects as keys and calculate them on their
    definition on the fly. All the keys are actually strings, but the AFrame modifies pd.DataFrame methods so
    they handle the conversion to strings and possibly creating the columns internally.

    So most of pd.DataFrame methods are actually modified via the metaclass - in particular, AColumns in *args and
    **kwargs of method calls are converted to strings and the columns are created if they don't exist. In addition,
    if the return is a pd.DataFrame, it is converted to AFrame.
    """
    def __init__(self, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def add_acolumn(self, acol: AColumn):
        """ Generate `acol AColumn` in the `AFrame`. """
        if not acol.name in self.columns:
            if self.verbose:
                print(f'Key "{acol}" not found in the AFrame, adding.')
            super().__setitem__(acol.name, acol.func(self))
            # self[acol] = acol.func(self)

    def __copy__(self, *args, **kwargs):
        """ Wrap the copied pd.DataFrame as AFrame. """
        return AFrame(super().__copy__(*args, **kwargs))

    def __deepcopy__(self, *args, **kwargs):
        """ Wrap the deepcopied pd.DataFrame as AFrame. """
        return AFrame(super().__deepcopy__(*args, **kwargs))

    def copy(self, *args, **kwargs):
        return AFrame(super().copy(*args, **kwargs))


class AFrameGroupBy(pd.core.groupby.generic.DataFrameGroupBy, metaclass=AMeta):
    """AFrame objects support also groupby operations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
