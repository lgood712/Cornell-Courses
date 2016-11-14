import copy
import itertools

def flatten(iterable):
    return list(itertools.chain.from_iterable(iterable))

def split(lst, num_chunks):
    chunk_size = int(len(lst) / num_chunks)
    chunks = (lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size))
    return chunks

class cached_property(object):
    """A cached property that recomputes every time its dependencies change

    Note that dependencies are checked for equality upon every call to
    the decorated property. If cheking equality is computationally intensive,
    you may be better off not listing the dependency, but instead manually
    deleting the cached value every time you change the dependency.
    """
    def __init__(self, dependencies=[]):
        self.dependencies = dependencies

    def __call__(self, func, doc=None):
        # called once per instantiation of the decorator
        self.func = func
        self.__doc__ = doc or func.__doc__
        self.__name__ = func.__name__
        self.__module__ = func.__module__
        return self

    def __get__(self, obj, cls):
        try:
            value = obj._cache[self.__name__]

            if self.dependencies:
                # confirm that dependencies haven't changed
                old_vals = self.dependency_vals
                new_vals = [getattr(obj, dep) for dep in self.dependencies]
                try:
                    if not all([old == current for old, current in zip(old_vals, new_vals)]):
                        raise AttributeError
                except ValueError:
                    # from numpy or pandas, truth value of array is ambiguous
                    if not all([(old == current).all().all().all().all().all()  # max 5 dimensions
                               for old, current in zip(old_vals, new_vals)]):
                        raise AttributeError
        except (KeyError, AttributeError, ValueError) :
            # value must be (re)computed
            value = self.func(obj)
            try:
                cache = obj._cache
            except AttributeError:
                cache = obj._cache = {}
            
            cache[self.__name__] = value
            self.dependency_vals = [copy.deepcopy(getattr(obj, dep)) for dep in self.dependencies]

        return value





