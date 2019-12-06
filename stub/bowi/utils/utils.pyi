from typing import Any, Callable, Hashable, List, Optional, Tuple, Type, TypeVar

T = TypeVar('T')
logger: Any
T_has = TypeVar('T_has', bound=Hashable)

def ignore_exception(func: Callable[..., T], exceptions: Tuple[Type[Exception]]) -> Callable[..., Optional[T]]: ...
def uniq(orig: List[T_has], lst: Optional[List[T_has]]=...) -> List[T_has]: ...
