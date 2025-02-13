import os
from typing import Dict, Any, List, Optional
import requests
import time
from datetime import datetime, timedelta

class ElfaAPI:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the ELFA API client.
        
        Args:
            api_key: API key for authentication. If not provided, will look for ELFA_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv('ELFA_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through ELFA_API_KEY environment variable")
            
        self.base_url = "https://api.elfa.ai/v1"
        self.headers = {
            "x-elfa-api-key": self.api_key,
            "Accept": "application/json"
        }

    def _make_request(self, endpoint: str, method: str = "GET", params: Dict = None) -> Dict:
        """Make a request to the ELFA API.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method to use
            params: Query parameters to include
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {str(e)}")
            raise

    def ping(self) -> bool:
        """Test API connectivity.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            response = self._make_request("ping")
            return response.get('success', False)
        except:
            return False

    def get_key_status(self) -> Dict[str, Any]:
        """Get current API key status and usage information.
        
        Returns:
            Dictionary containing key status information
        """
        return self._make_request("key-status")

    def search_mentions(
        self,
        keywords: List[str],
        days_ago: int = 30,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Search for mentions of specific keywords.
        
        Args:
            keywords: List of keywords to search for
            days_ago: Number of days to look back
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # Calculate timestamps
        end_time = int(time.time())
        start_time = int((datetime.now() - timedelta(days=days_ago)).timestamp())
        
        # Format keywords
        formatted_keywords = ','.join(keywords)
        
        params = {
            'keywords': formatted_keywords,
            'from': start_time,
            'to': end_time,
            'limit': limit
        }
        
        return self._make_request("mentions/search", params=params)

    def get_mentions(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get recent mentions.
        
        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            Dictionary containing mentions
        """
        params = {
            'limit': limit,
            'offset': offset
        }
        return self._make_request("mentions", params=params)

    def get_top_mentions(
        self,
        ticker: str,
        time_window: str = "7d",
        page: int = 1,
        page_size: int = 50,
        include_account_details: bool = False
    ) -> Dict[str, Any]:
        """Get top mentions for a specific ticker.
        
        Args:
            ticker: Ticker symbol (e.g. "$HEU")
            time_window: Time window to look at (e.g. "7d")
            page: Page number for pagination
            page_size: Number of results per page
            include_account_details: Whether to include account details
            
        Returns:
            Dictionary containing top mentions
        """
        params = {
            'ticker': ticker,
            'timeWindow': time_window,
            'page': page,
            'pageSize': page_size,
            'includeAccountDetails': str(include_account_details).lower()
        }
        return self._make_request("top-mentions", params=params)

    def get_trending_tokens(
        self,
        time_window: str = "24h",
        page: int = 1,
        page_size: int = 50,
        min_mentions: int = 5
    ) -> Dict[str, Any]:
        """Get trending tokens.
        
        Args:
            time_window: Time window to look at (e.g. "24h")
            page: Page number for pagination
            page_size: Number of results per page
            min_mentions: Minimum number of mentions required
            
        Returns:
            Dictionary containing trending tokens
        """
        params = {
            'timeWindow': time_window,
            'page': page,
            'pageSize': page_size,
            'minMentions': min_mentions
        }
        return self._make_request("trending-tokens", params=params)

    def get_account_smart_stats(self, username: str) -> Dict[str, Any]:
        """Get smart statistics for a specific account.
        
        Args:
            username: Twitter username
            
        Returns:
            Dictionary containing account statistics
        """
        params = {'username': username}
        return self._make_request("account/smart-stats", params=params)