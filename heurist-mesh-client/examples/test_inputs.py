TOOL_TEST_INPUTS = {
    "AaveAgent": {"get_aave_reserves": {"chain_id": 137}},
    "AlloraPricePredictionAgent": {"get_allora_prediction": {"token": "ETH", "timeframe": "5m"}},
    "BitquerySolanaTokenInfoAgent": {
        "query_token_metrics": {"token_address": "So11111111111111111111111111111111111111112", "quote_token": "sol"},
        "query_token_holders": {"token_address": "So11111111111111111111111111111111111111112"},
        "query_token_buyers": {"token_address": "So11111111111111111111111111111111111111112", "limit": 30},
        "query_top_traders": {"token_address": "So11111111111111111111111111111111111111112", "limit": 30},
        "query_holder_status": {
            "token_address": "So11111111111111111111111111111111111111112",
            "buyer_addresses": [
                "ApRJBQEKfmcrViQkH94BkzRFUGWtA8uC71DXu6USdd3n",
                "9nG4zw1jVJFpEtSLmbGQpTnpG2TiKfLXWkkTyyRvxTt6",
            ],
        },
        "get_top_trending_tokens": {},
    },
    "CarvOnchainDataAgent": {
        "query_onchain_data": {
            "blockchain": "solana",
            "query": "What's the most active address on Bitcoin during the last 24 hours?",
        }
    },
    "CoinGeckoTokenInfoAgent": {
        "get_coingecko_id": {"token_name": "ethereum"},
        "get_token_info": {"coingecko_id": "ethereum"},
        "get_trending_coins": {},
        "get_token_price_multi": {
            "ids": "bitcoin,ethereum,solana",
            "vs_currencies": "usd",
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True,
        },
        "get_categories_list": {},
        "get_category_data": {"order": "market_cap_desc"},
        "get_tokens_by_category": {
            "category_id": "layer-1",
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 10,
            "page": 1,
        },
    },
    "DeepResearchAgent": {"deep_research": {"query": "What is the total value locked in Aave v3?"}},
    "DexScreenerTokenInfoAgent": {
        "search_pairs": {"search_term": "ETH"},
        "get_specific_pair_info": {
            "chain": "base",
            "pair_address": "0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a",
        },
        "get_token_pairs": {"chain": "solana", "token_address": "8TE8oxirpnriy9CKCd6dyjtff2vvP3n6hrSMqX58pump"},
    },
    "DuckDuckGoSearchAgent": {"search_web": {"search_term": "latest crypto regulations"}},
    "ElfaTwitterIntelligenceAgent": {
        "search_mentions": {"keywords": ["aave"]},
        "search_account": {"username": "aave"},
        "get_trending_tokens": {},
    },
    "ExaSearchAgent": {
        "exa_web_search": {"search_term": "latest defi trends"},
        "exa_answer_question": {"question": "What is the current state of DeFi?"},
    },
    "FirecrawlSearchAgent": {
        "firecrawl_web_search": {"search_term": "What are the latest developments in zero knowledge proofs?"},
        "firecrawl_extract_web_data": {
            "urls": ["https://ethereum.org/en/zero-knowledge-proofs/"],
            "extraction_prompt": "Extract information about how zero knowledge proofs are being used in blockchain technology",
        },
    },
    "FundingRateAgent": {
        "get_all_funding_rates": {},
        "get_symbol_funding_rates": {"symbol": "BTC"},
        "find_cross_exchange_opportunities": {},
        "find_spot_futures_opportunities": {},
    },
    "GoplusAnalysisAgent": {
        "fetch_security_details": {"contract_address": "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0"}
    },
    "MasaTwitterSearchAgent": {"search_twitter": {"search_term": "defi"}},
    "MetaSleuthSolTokenWalletClusterAgent": {
        "fetch_token_clusters": {"address": "tQNVaFm2sy81tWdHZ971ztS5FKaShJUKGAzHMcypump"},
        "fetch_cluster_details": {"cluster_uuid": "13axGrDoFlaj8E0ruhYfi1"},
    },
    "PumpFunTokenAgent": {
        "query_recent_token_creation": {},
        "query_latest_graduated_tokens": {"timeframe": 48},
    },
    "SolWalletAgent": {
        "analyze_common_holdings_of_top_holders": {"token_address": "J7tYmq2JnQPvxyhcXpCDrvJnc9R5ts8rv7tgVHDPsw7U"},
        "get_tx_history": {"owner_address": "DbDi7soBXALYRMZSyJMEAfpaK3rD1hr5HuCYzuDrcEEN"},
        "get_wallet_assets": {"owner_address": "DbDi7soBXALYRMZSyJMEAfpaK3rD1hr5HuCYzuDrcEEN"},
    },
    "ZerionWalletAnalysisAgent": {
        "fetch_wallet_tokens": {"wallet_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D"},
        "fetch_wallet_nfts": {"wallet_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D"},
    },
    "TwitterInsightAgent": {
        "get_smart_followers_history": {"username": "heurist_ai", "timeframe": "D7"},
        "get_smart_followers_categories": {"username": "heurist_ai"},
        "get_smart_mentions_feed": {"username": "heurist_ai", "limit": 100},
    },
}
