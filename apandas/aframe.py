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

        # just for debugging
        def _print(*args, **kwargs):
            # print(*args, **kwargs)
            return None

        def any_acol_in_tree(t):
            flags = []
            def acol_in_tree(t):
                b = isinstance(t, AColumn) or (isinstance(t, dict) and any(isinstance(k, AColumn) for k in t.keys()))
                flags.append(b)
            tree.traverse(acol_in_tree, t)
            return any(flags)

        def method_wrapper(method, map_args=True):
            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                if map_args:
                    orig_args = args.__copy__() if hasattr(args, '__copy__') else args
                    orig_kwargs = kwargs.copy()

                    def map_keys(mapping):
                        if isinstance(mapping, dict):
                            keys = list(mapping.keys())
                            for key in keys:
                                if isinstance(key, AColumn):
                                    if method.__name__ != '__setitem__' and key.func is not None:
                                        if name == 'AFrame':
                                            self.add_acolumn(key)
                                        elif name == 'AFrameGroupBy':
                                            self.obj.add_acolumn(key)
                                    mapping[key.name] = mapping.pop(key)

                    def map_leaves(x):
                        if isinstance(x, AColumn):
                            if method.__name__ != '__setitem__' and x.func is not None:
                                if name == 'AFrame':
                                    self.add_acolumn(x)
                                elif name == 'AFrameGroupBy':
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

                # if method.__name__ not in ['__len__', '__repr__', 'to_string', '__getattr__']:
                #     print(f'Calling {method.__name__} on {self.__class__.__name__} with args={tuple(a.__repr__() for a in args)}, kwargs={kwargs}')
                #     # if self.__class__.__name__ == 'AFrame':
                #     #     print('Here is the full frame\n', self)

                result = method(self, *args, **kwargs)
                if isinstance(result, pd.DataFrame) and not isinstance(result, AFrame):
                    # print(f'Converting pd.DataFrame {result} to an AFrame:')
                    result = AFrame(result)
                elif isinstance(result, pd.core.groupby.generic.DataFrameGroupBy) and not isinstance(
                        result, AFrameGroupBy):
                    if isinstance(self, AFrame):
                        result = AFrameGroupBy(self, *args, **kwargs)
                    elif isinstance(self, AFrameGroupBy) and method.__name__ == '__getitem__':
                        result = AFrameGroupBy(self.obj, keys=self.keys, axis=self.axis, as_index=self.as_index,
                                               selection=args[0], group_keys=self.group_keys, dropna=self.dropna,
                                               grouper=self.grouper, exclusions=self.exclusions)
                elif isinstance(result, pd.Series) and not isinstance(result, ASeries):
                    result = ASeries(result)
                elif isinstance(result, pd.core.groupby.generic.SeriesGroupBy) and not isinstance(
                        result, ASeriesGroupBy):
                    if isinstance(self, ASeries):
                        result = ASeriesGroupBy(self, *args, **kwargs)
                    elif isinstance(self, AFrameGroupBy) and method.__name__ == '__getitem__':
                        # Series groupby does not use some properties and have not to be propagated
                        result = ASeriesGroupBy(self.obj[args[0]], selection=args[0], dropna=self.dropna,
                                                keys=None, grouper=self.grouper)

                return result
            return wrapper

        if name in ['ASeriesGroupBy', 'AFrameGroupBy']:
            # for GroupBys, process also parent's parent
            for attr_name, attr_value in parent_class.__bases__[0].__dict__.items():
                if attr_name not in aclass.__dict__ and callable(attr_value):
                    # if not attr_name.startswith('_') or (attr_name == '__getitem__' and name == 'AFrameGroupBy'):
                    if not attr_name.startswith('_') or (attr_name.startswith('__') and attr_name.endswith('__')):
                        _print(f'Wrapping the output for {attr_name} on {name} '
                                  f'from {parent_class.__bases__[0].__name__}')
                        setattr(aclass, attr_name, method_wrapper(attr_value, map_args=False))
                    else:
                        _print(f'Keeping for {attr_name} on {name} from {parent_class.__bases__[0].__name__}')
                        setattr(aclass, attr_name, attr_value)

        for attr_name, attr_value in parent_class.__dict__.items():
            if attr_name not in aclass.__dict__ and callable(attr_value):
                # if not attr_name.startswith('_') or (attr_name == '__getitem__' and name == 'AFrameGroupBy'):
                if not attr_name.startswith('_') or attr_name in ['__getitem__', '__setitem__']:
                    _print(f'Modifying {attr_name} on {name} from {parent_class.__name__}')
                    setattr(aclass, attr_name, method_wrapper(attr_value))
                elif attr_name.startswith('__') and attr_name.endswith('__'):
                    _print(f'Wrapping the output for {attr_name} on {name} from {parent_class.__name__}')
                    setattr(aclass, attr_name, method_wrapper(attr_value, map_args=False))
                else:
                    _print(f'Keeping for {attr_name} on {name} from {parent_class.__name__}')
                    setattr(aclass, attr_name, attr_value)
        return aclass


class ASeries(pd.Series, metaclass=AMeta):
    """
    ASeries is a subclass of pd.Series that allows to use AColumn in renaming and returns AFrame when converted
    to a dataframe (for instance via `.reset_index` method).
    """
    def to_pandas(self):
        """ Unwrap to pd.Series. """
        return pd.Series(self)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __copy__(self, *args, **kwargs):
        """ Wrap the copied pd.Series as ASeries. """
        return ASeries(super().__copy__(*args, **kwargs))

    def __deepcopy__(self, *args, **kwargs):
        """ Wrap the deepcopied pd.Series as ASeries. """
        return ASeries(super().__deepcopy__(*args, **kwargs))

    def copy(self, *args, **kwargs):
        """ Wrap the copied pd.Series as ASeries. """
        return ASeries(super().copy(*args, **kwargs))


class ASeriesGroupBy(pd.core.groupby.generic.SeriesGroupBy, metaclass=AMeta):
    """ASeries objects support also groupby operations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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

    def to_pandas(self):
        """ Unwrap to pd.DataFrame. """
        return pd.DataFrame(self)

    def __copy__(self, *args, **kwargs):
        """ Wrap the copied pd.DataFrame as AFrame. """
        return AFrame(super().__copy__(*args, **kwargs))

    def __deepcopy__(self, *args, **kwargs):
        """ Wrap the deepcopied pd.DataFrame as AFrame. """
        return AFrame(super().__deepcopy__(*args, **kwargs))

    def copy(self, *args, **kwargs):
        """ Wrap the copied pd.DataFrame as AFrame. """
        return AFrame(super().copy(*args, **kwargs))


class AFrameGroupBy(pd.core.groupby.generic.DataFrameGroupBy, metaclass=AMeta):
    """AFrame objects support also groupby operations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
