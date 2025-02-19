import os
import json
import requests
import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from .mesh_agent import MeshAgent, with_cache, with_retry, monitor_execution
from core.llm import call_llm_async, call_llm_with_tools_async

class MoniTwitterSmartAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update({
            'name': 'Moni Twitter Smart Agent',
            'version': '1.0.0',
            'author': 'Heurist team',
            'author_address': '0x7d9d1821d15B9e0b8Ab98A058361233E255E405D',
            'description': 'This agent provides Twitter analytics data by leveraging the Moni API endpoints.',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'Natural language query about a Twitter account analytics request',
                    'type': 'str',
                    'required': True
                },
                {
                    'name': 'raw_data_only',
                    'description': 'If true, the agent returns only the raw API result data',
                    'type': 'bool',
                    'required': False,
                    'default': False
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'A natural language explanation of the fetched Twitter analytics data',
                    'type': 'str'
                },
                {
                    'name': 'data',
                    'description': 'Structured data returned by the API call',
                    'type': 'dict'
                }
            ],
            'external_apis': ['Moni'],
            'tags': ['Twitter', 'Analytics', 'Social']
        })

    def get_system_prompt(self) -> str:
        return (
            "You are a specialized assistant that analyzes Twitter analytics data by querying the Moni API. "
            "When requested, you should fetch data such as smart followers, their distribution, category information, "
            "timeline tweets, or start tweet tracking and then provide a concise, data-driven summary."
        )

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'get_smart_followers',
                    'description': 'Retrieve the smart followers for a given Twitter account',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'username': {
                                'type': 'string',
                                'description': 'Twitter username'
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Maximum number of followers to return',
                                'default': 100
                            },
                            'offset': {
                                'type': 'integer',
                                'description': 'Starting index for followers',
                                'default': 0
                            },
                            'changesTimeframe': {
                                'type': 'string',
                                'description': 'Timeframe for showing changes (H1, H24, D7, D30, Y1)',
                                'default': 'H24'
                            },
                            'orderBy': {
                                'type': 'string',
                                'description': 'Order of the results',
                                'default': ''
                            }
                        },
                        'required': ['username']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_smart_followers_distribution',
                    'description': 'Retrieve the distribution of smart followers by level for a Twitter account',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'username': {
                                'type': 'string',
                                'description': 'Twitter username'
                            },
                            'timeframe': {
                                'type': 'string',
                                'description': 'Timeframe for the distribution (H1, H24, D7, D30, Y1)',
                                'default': 'H24'
                            }
                        },
                        'required': ['username']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_smart_followers_categories',
                    'description': 'Retrieve the smart followers categories for a Twitter account',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'username': {
                                'type': 'string',
                                'description': 'Twitter username'
                            },
                            'timeframe': {
                                'type': 'string',
                                'description': 'Timeframe for the categories (H1, H24, D7, D30, Y1)',
                                'default': 'H24'
                            }
                        },
                        'required': ['username']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_timeline',
                    'description': 'Retrieve the timeline of tweets for a Twitter account',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'username': {
                                'type': 'string',
                                'description': 'Twitter username'
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Maximum number of tweets to return',
                                'default': 100
                            },
                            'offset': {
                                'type': 'integer',
                                'description': 'Starting index of the first tweet',
                                'default': 0
                            }
                        },
                        'required': ['username']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'start_tweet_tracking',
                    'description': 'Start tweets tracking for a Twitter account',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'username': {
                                'type': 'string',
                                'description': 'Twitter username'
                            }
                        },
                        'required': ['username']
                    }
                }
            }
        ]

    async def get_smart_followers(self, username: str, limit: int = 100, offset: int = 0,
                                    changesTimeframe: str = "H24", orderBy: str = "") -> Dict:
        try:
            url = f"https://api.discover.getmoni.io/api/v1/twitters/{username}/smart_followers/"
            params = {
                'limit': limit,
                'offset': offset,
                'changesTimeframe': changesTimeframe
            }
            if orderBy:
                params['orderBy'] = orderBy
            headers = {"Api-Key": os.getenv("MONI_API_KEY")}
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch smart followers: {str(e)}"}

    async def get_smart_followers_distribution(self, username: str, timeframe: str = "H24") -> Dict:
        try:
            url = f"https://api.discover.getmoni.io/api/v1/twitters/{username}/smart_followers/distribution/level/"
            params = {'timeframe': timeframe}
            headers = {"Api-Key": os.getenv("MONI_API_KEY")}
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch smart followers distribution: {str(e)}"}

    async def get_smart_followers_categories(self, username: str, timeframe: str = "H24") -> Dict:
        try:
            url = f"https://api.discover.getmoni.io/api/v1/twitters/{username}/smart_followers/categories/"
            params = {'timeframe': timeframe}
            headers = {"Api-Key": os.getenv("MONI_API_KEY")}
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch smart followers categories: {str(e)}"}

    async def get_timeline(self, username: str, limit: int = 100, offset: int = 0) -> Dict:
        try:
            url = f"https://api.discover.getmoni.io/api/v1/twitters/{username}/timeline/"
            params = {
                'limit': limit,
                'offset': offset
            }
            headers = {"Api-Key": os.getenv("MONI_API_KEY")}
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch timeline: {str(e)}"}

    async def start_tweet_tracking(self, username: str) -> Dict:
        try:
            url = f"https://api.discover.getmoni.io/api/v1/twitters/{username}/tracker/tweet/"
            headers = {"Api-Key": os.getenv("MONI_API_KEY")}
            response = requests.post(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
            # No response body expected
            return {"status": "tracking started"}
        except Exception as e:
            return {"error": f"Failed to start tweet tracking: {str(e)}"}

    async def handle_message(self, params: Dict) -> Dict:
        query = params.get('query', '')
        print("query", query)
        # Call LLM to decide which tool (function) to use.
        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            system_prompt=self.get_system_prompt(),
            user_prompt=query,
            tools=self.get_tool_schemas(),
            temperature=0.6
        )
        if not response:
            return {"error": "Failed to process query"}

        if not response.get('tool_calls'):
            return {"response": response['content'], "data": {}}

        tool_call = response['tool_calls']
        function_args = json.loads(tool_call.function.arguments)


        if tool_call.function.name == 'get_smart_followers':
            username = function_args.get('username')
            limit = function_args.get('limit', 100)
            offset = function_args.get('offset', 0)
            changesTimeframe = function_args.get('changesTimeframe', "H24")
            orderBy = function_args.get('orderBy', "")
            data = await self.get_smart_followers(username, limit, offset, changesTimeframe, orderBy)
        elif tool_call.function.name == 'get_smart_followers_distribution':
            username = function_args.get('username')
            timeframe = function_args.get('timeframe', "H24")
            data = await self.get_smart_followers_distribution(username, timeframe)
        elif tool_call.function.name == 'get_smart_followers_categories':
            username = function_args.get('username')
            timeframe = function_args.get('timeframe', "H24")
            data = await self.get_smart_followers_categories(username, timeframe)
        elif tool_call.function.name == 'get_timeline':
            username = function_args.get('username')
            limit = function_args.get('limit', 100)
            offset = function_args.get('offset', 0)
            data = await self.get_timeline(username, limit, offset)
        elif tool_call.function.name == 'start_tweet_tracking':
            username = function_args.get('username')
            data = await self.start_tweet_tracking(username)
        else:
            return {"error": "Unsupported function"}

        raw_data_only = params.get('raw_data_only', False)
        if raw_data_only:
            return {"response": "", "data": data}

        # Generate an explanation using the LLM based on the fetched data.
        explanation = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
               {"role": "tool", "content": str(data), "tool_call_id": tool_call.id}
            ],
            temperature=0.5
        )
        return {"response": explanation, "data": data} 