"""robots.txt checking. If a site disallows a path, we do not fetch it."""
from __future__ import annotations

from urllib import robotparser
from urllib.parse import urlparse


class RobotsCache:
    def __init__(self, user_agent: str):
        self.ua = user_agent
        self._parsers: dict[str, robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if base not in self._parsers:
            rp = robotparser.RobotFileParser(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None  # unreachable robots.txt: stay conservative below
            self._parsers[base] = rp
        rp = self._parsers[base]
        if rp is None:
            return False
        return rp.can_fetch(self.ua, url)
