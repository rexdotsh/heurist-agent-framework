import logging
from typing import Any, Optional

import aiohttp
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class BaseAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 10
        self.session = requests.Session()
        self.async_session: Optional[aiohttp.ClientSession] = None

    def _sync_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Simple synchronous request"""
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        try:
            response = self.session.request(method, f"{self.base_url}{endpoint}", **kwargs)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    async def _async_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Async request"""
        if not self.async_session:
            self.async_session = aiohttp.ClientSession()

        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        try:
            async with getattr(self.async_session, method.lower())(f"{self.base_url}{endpoint}", **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Async API request failed: {e}")
            raise

    async def close(self):
        if self.async_session:
            await self.async_session.close()
            self.async_session = None

    def __del__(self):
        self.session.close()
