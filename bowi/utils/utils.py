import logging
from typing import Callable, Hashable, List, Optional, Tuple, Type, TypeVar


T = TypeVar('T')
logger = logging.getLogger(__file__)
T_has = TypeVar('T_has', bound=Hashable)


def ignore_exception(func: Callable[..., T],
                     exceptions: Tuple[Type[Exception]]) -> Callable[..., Optional[T]]:
    """
    A decorator to ignore certain exceptions
    """
    def wrapper(*args, **kwargs) -> Optional[T]:
        try:
            val: T = func(*args, **kwargs)
            return val
        except exceptions as e:
            logger.error(e, exc_info=True)
            return None
    return wrapper


def uniq(orig: List[T_has],
         lst: Optional[List[T_has]] = None) -> List[T_has]:
    """
    It works as a uniq command. This asserts orig is sorted.
    """
    if lst is None:
        return uniq(orig, [])
    elif len(orig) == 0:
        return lst
    else:
        head, *tail = orig
        if head in lst:
            return uniq(orig=tail, lst=lst)
        else:
            return uniq(orig=tail, lst=[head] + lst)
