"""Expose the local TextBlob shim regardless of sys.path."""

from AIWaifu.textblob import *  # type: ignore[F401,F403]
