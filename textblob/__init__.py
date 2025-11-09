"""Expose the local TextBlob shim regardless of sys.path."""

from AIWaifu import textblob as _textblob

__all__ = getattr(_textblob, "__all__", [])

globals().update({name: getattr(_textblob, name) for name in __all__})
