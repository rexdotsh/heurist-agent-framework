from typing import Dict, Any, List, Optional
from .mesh_agent import MeshAgent, monitor_execution, with_retry, with_cache
import logging
from clients.defillama_client import DefiLlamaClient
from clients.merkl_client import MerklClient
import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, tool, Model, ChatMessage, Tool
from smolagents.models import parse_tool_args_if_needed
from smolagents.agents import populate_template
from smolagents.memory import SystemPromptStep
from core.custom_smolagents import smolagents_system_prompt, OpenAIServerModel
import asyncio
import openai

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def is_zk_rewards(item):
    """Check if an opportunity has ZK token rewards"""
    ZK_TOKEN_ADDRESS = "0x5A7d6b2F92C77FAD6CCaBd7EE0624E64907Eaf3E".lower()
    
    rewards_record = item.get('rewardsRecord', {})
    breakdowns = rewards_record.get('breakdowns', [])
    
    for breakdown in breakdowns:
        token = breakdown.get('token', {})
        if (
            token.get('name') == 'ZKsync' or 
            token.get('address', '').lower() == ZK_TOKEN_ADDRESS
        ):
            return True
    return False

def get_zkignite_overview(client: MerklClient):
    @tool
    def _get_zkignite_overview() -> Dict[str, Any]:
        """Get an overview of DeFi yield farming opportunities in zkIgnite program. Return a list of opportunities including protocol name, opportunity name, APR, TVL, and data ready for creating a chart"""
        data = []
        chart_data = []
        baseData = client.get_opportunities(chain_id="324", items=100)
        
        for item in baseData:
            if 'protocol' not in item or item['status'] != 'LIVE' or not is_zk_rewards(item):
                continue
                
            data.append({
                'protocol_name': item['protocol']['name'],
                'opportunity_name': item['name'],
                'apr': item['apr'],
                'tvl': item['tvl']
            })
            
            # Format chart data
            tokens = [{'id': t['id'], 'icon': t['icon']} for t in item.get('tokens', [])]
            protocol = {
                'name': item['protocol']['name'],
                'icon': item['protocol']['icon']
            }
            
            chart_data.append({
                'name': item['name'].replace('Provide liquidity to ', '')
                              .replace('Supply ', '')
                              .replace(' ', '\n'),
                'protocol': protocol,
                'tvl': item['tvl'],
                'apr': item['apr'],
                'dailyRewards': item['dailyRewards'],
                'tokens': tokens
            })
        
        logger.info("_get_zkignite_overview done")
        return {
            "opportunities": data,
            # TODO: Token URL is too long for the model to handle
            #"chart_data": chart_data
        }
    return _get_zkignite_overview

def get_protocol_opportunities(client: MerklClient):
    @tool
    def _get_protocol_opportunities(protocol_ids: List[str]) -> Dict[str, Any]:
        """Get yield opportunities for specific protocols from the zkIgnite program

Args:
    protocol_ids: List of protocol IDs (e.g. ['aave', 'syncswap'])
    Available protocols: aave, holdstation, izumi, koi, maverick, pancakeswap-v3,
    reactorfusion, rfx, syncswap, uniswap-v3, venus, vestexchange, woofi, zerolend, zkswap
"""
        protocol_ids_str = ",".join(protocol_id.strip().lower() for protocol_id in protocol_ids)
        
        responses = client.get_opportunities(
            chain_id="324",
            main_protocol_id=protocol_ids_str,
            items=5*len(protocol_ids),
            status='LIVE'
        )
        
        opportunities = []
        for item in responses:
            if not is_zk_rewards(item):
                continue
                
            opportunities.append({
                "name": item["name"],
                "tvl": item["tvl"],
                "apr": item["apr"],
                "dailyRewards": item["dailyRewards"],
                "status": item["status"]
            })
            
        logger.info("_get_protocol_opportunities done")
        return {
            "protocol_id": protocol_ids,
            "opportunities": opportunities
        }
    return _get_protocol_opportunities

def get_top_yield_opportunities(client: MerklClient):
    @tool
    def _get_top_yield_opportunities(top: int = 10, category: Optional[str] = None) -> Dict[str, Any]:
        """Get top yield opportunities filtered by token category

Args:
    top: Number of opportunities to return (default: 10)
    category: Filter by token category:
        - "zk": ZK token related pools
        - "stable": Stablecoin pools (USDT, USDC, USDC.e)
        - "eth": ETH related tokens (ETH, WETH, etc)
        - None: No filter
    """
        token_filters = {
            "zk": ["ZK"],
            "stable": ["USDT", "USDC", "USDC.e"],
            "eth": ["ETH", "WETH", "wstETH", "wrsETH"]
        }

        responses = client.get_opportunities(
            chain_id="324",
            items=100,
            sort="apr",
            order="desc"
        )
        
        data = []
        for item in responses:
            if 'protocol' not in item or item['status'] != 'LIVE' or not is_zk_rewards(item):
                continue
                
            if category in token_filters:
                matches_category = False
                for token in item['tokens']:
                    if any(symbol in token['symbol'] for symbol in token_filters[category]):
                        matches_category = True
                        break
                if not matches_category:
                    continue
                    
            data.append({
                'protocol_name': item['protocol']['name'],
                'opportunity_name': item['name'],
                'apr': item['apr'],
                'tvl': item['tvl'],
                'tokens': [t['symbol'] for t in item['tokens']]
            })
            
        data.sort(key=lambda x: x['apr'], reverse=True)
        logger.info("_get_top_yield_opportunities done")
        return {
            "opportunities": data[:top]
        }
    return _get_top_yield_opportunities

DEFILLAMA_PROTOCOL_NAME_TO_INFO = {
    "Aave": {
        "id": "aave",
        "tag": "Lending",
        "parentProtocol": "parent#aave"
    },
    "Holdstation": {
        "id": "holdstation",
        "tag": "Perpetuals",
        "parentProtocol": "parent#holdstation"
    },
    "Izumi": {
        "id": "izumi-finance",
        "tag": "DEX",
        "parentProtocol": "parent#izumi-finance"
    },
    "Koi Finance": {
        "id": "koi-finance",
        "tag": "DEX",
        "parentProtocol": "parent#koi-finance"
    },
    "Maverick": {
        "id": "maverick-protocol",
        "tag": "DEX",
        "parentProtocol": "parent#maverick-protocol"
    },
    "PancakeSwap": {
        "id": "pancakeswap",
        "tag": "DEX",
        "parentProtocol": "parent#pancakeswap"
    },
    "Reactor Fusion": {
        "id": "reactorfusion",
        "tag": "Lending",
        "parentProtocol": "2880",
    },
    "RFX": {
        "id": "rfx-exchange",
        "parentProtocol": "5406",
        "tag": "Perpetuals"
    },
    "SyncSwap": {
        "id": "syncswap",
        "parentProtocol": "2728",
        "tag": "DEX"
    },
    "Uniswap": {
        "id": "uniswap-v3",
        "parentProtocol": "parent#uniswap",
        "tag": "DEX"
    },
    "Venus": {
        "id": "venus",
        "parentProtocol": "parent#venus-finance",
        "tag": "Lending"
    },
    "Vest Exchange": {
        "id": "vest-exchange",
        "parentProtocol": "4400",
        "tag": "Perpetuals"
    },
    "WOOFi": {
        "id": "woofi",
        "parentProtocol": "parent#woofi",
        "tag": "DEX"
    },
    "ZeroLend": {
        "id": "zerolend",
        "parentProtocol": "3243",
        "tag": "Lending"
    },
    "ZKSwap": {
        "id": "zkswap-finance",
        "parentProtocol": "parent#zkswap-finance",
        "tag": "DEX"
    }
}

def get_tvl_overview(client: DefiLlamaClient):
    @tool
    def _get_tvl_overview() -> Dict[str, Any]:
        """Get TVL overview across zkSync DeFi protocols"""
        data = []
        protocols = client.get_protocols()
        
        # Create mapping for parent protocols
        parent_protocol_map = {}
        id_protocol_map = {}
        for protocol in protocols:
            if protocol.get('parentProtocol'):
                if protocol['parentProtocol'] not in parent_protocol_map:
                    parent_protocol_map[protocol['parentProtocol']] = []
                parent_protocol_map[protocol['parentProtocol']].append(protocol)
            if protocol.get('id'):
                id_protocol_map[protocol['id']] = protocol
        
        # Process each protocol in DEFILLAMA_PROTOCOL_NAME_TO_INFO
        for protocol_name, info in DEFILLAMA_PROTOCOL_NAME_TO_INFO.items():
            protocol_id = info['id']
            parent_protocol = info['parentProtocol']
            
            try:
                target_protocols = []
                if isinstance(parent_protocol, str) and parent_protocol.startswith('parent#'):
                    target_protocols = parent_protocol_map.get(parent_protocol, [])
                else:
                    if parent_protocol in id_protocol_map:
                        target_protocols = [id_protocol_map[parent_protocol]]

                tvl = 0
                for protocol in target_protocols:
                    chain_tvls = protocol.get('chainTvls', {})
                    zksync_tvl = chain_tvls.get('ZKsync Era', 0) or chain_tvls.get('zkSync Era', 0)
                    if zksync_tvl:
                        tvl += zksync_tvl

                data.append({
                    'name': protocol_name,
                    'tvl': tvl
                })
                
            except Exception as e:
                logger.error(f"Error processing protocol {protocol_name}: {str(e)}")
                continue
            
        return {
            "tvl_data": data
        }
    return _get_tvl_overview

class ZkIgniteAnalystAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self._merkl_client = MerklClient()
        self._defillama_client = DefiLlamaClient()
        
        self.metadata.update({
            'name': 'ZkIgnite Analyst Agent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'description': 'This agent analyzes zkSync Era DeFi opportunities in the zkIgnite program and has access to real-time yield and TVL data',
            'inputs': [{
                'name': 'query',
                'description': 'User query about zkSync DeFi opportunities or protocols',
                'type': 'str',
                'required': True
            }],
            'outputs': [{
                'name': 'response',
                'description': 'Analysis results',
                'type': 'str'
            }],
            'large_model_id': 'openai/gpt-4o-mini',
            'external_apis': ['Merkl', 'DefiLlama'],
            'tags': ['DeFi', 'Yield Farming', 'ZKsync', 'Reasoning']
        })
        
        tools = [
            get_zkignite_overview(self._merkl_client),
            get_protocol_opportunities(self._merkl_client),
            get_top_yield_opportunities(self._merkl_client),
            get_tvl_overview(self._defillama_client)
        ]

        # Use OpenRouter to access Gemini. Most open source models cannot handle the tool calling format.
        self.model = OpenAIServerModel(
            model_id=self.metadata['large_model_id'],
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        max_steps = 5
        self.agent = ToolCallingAgent(
            tools=tools,
            model=self.model,
            max_steps=max_steps
        )

        self.agent.prompt_templates["system_prompt"] = smolagents_system_prompt()
        self.agent.system_prompt = self.agent.prompt_templates["system_prompt"]
        self.agent.memory.system_prompt = SystemPromptStep(system_prompt=self.agent.system_prompt)

        logger.info(f"ZkIgnite Analyst Agent initialized. System prompt: {self.agent.system_prompt}")
        
        self.agent.step_callbacks.append(self._step_callback)
        
    def _step_callback(self, step_log):
        # TODO: push this to server
        print("step", step_log)
        if step_log.tool_calls:
            print(f"Calling function {step_log.tool_calls[0].name} with args {step_log.tool_calls[0].arguments}")
            
    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message with multi-tool analysis"""
        query = params.get('query')
        if not query:
            raise ValueError("Query parameter is required")
            
        try:
            result = self.agent.run(
                f"""Analyze this query and provide insights: {query}

Guidelines:
- Combine data from multiple tools when needed
- Format numbers clearly (e.g. $1.5M, 15.2%) without too many decimals
- Keep response concise and focused on key insights
"""
            )
            
            return {
                "response": result.to_string(),
                "reasoning_steps": self.agent.memory.get_succinct_steps()
            }
                
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return {"error": str(e)}
