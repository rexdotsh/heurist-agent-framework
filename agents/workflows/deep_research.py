from typing import Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResearchQuery:
    query: str
    research_goal: str

class ResearchResult(TypedDict):
    learnings: List[str]
    visited_urls: List[str]
    follow_up_questions: List[str]

class ResearchWorkflow:
    """Research workflow combining interactive and autonomous research patterns"""
    
    def __init__(self, llm_provider, tool_manager, firecrawl_client):
        self.llm_provider = llm_provider
        self.tool_manager = tool_manager
        self.firecrawl = firecrawl_client
        
    async def process(
        self,
        message: str,
        personality_provider=None,
        chat_id: str = None,
        workflow_options: Dict = None,
        **kwargs
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Main research workflow processor"""
        
        # Set default options
        options = {
            "interactive": False,  # Whether to ask clarifying questions first
            "breadth": 3,  # Number of parallel searches
            "depth": 2,  # How deep to go in research
            "concurrency": 3,  # Max concurrent requests
            "temperature": 0.7,
        }
        
        if workflow_options:
            options.update(workflow_options)
            
        try:
            if options["interactive"]:
                # Interactive research flow with clarifying questions
                questions = await self._generate_questions(message)
                # Here we'd typically wait for user response, but for now we'll proceed
                enhanced_query = f"{message}\nConsidering questions: {', '.join(questions)}"
            else:
                enhanced_query = message
                
            # Conduct deep research
            research_result = await self._deep_research(
                query=enhanced_query,
                breadth=options["breadth"],
                depth=options["depth"],
                concurrency=options["concurrency"]
            )
            
            # Generate final report
            report = await self._generate_report(
                original_query=message,
                research_result=research_result,
                personality_provider=personality_provider
            )
            
            return report, None, research_result
            
        except Exception as e:
            logger.error(f"Research workflow failed: {str(e)}")
            return f"Research failed: {str(e)}", None, None
            
    async def _generate_questions(self, query: str) -> List[str]:
        """Generate clarifying questions for research"""
        prompt = f"""Given this research topic: {query}, generate 3-5 follow-up questions to better understand the research needs.
        Return ONLY a JSON array of strings containing the questions."""
        
        response = await self.llm_provider.call(
            system_prompt="You are a research assistant helping to clarify research queries.",
            user_prompt=prompt,
            temperature=0.7
        )
        
        try:
            questions = json.loads(response["content"])
            return questions if isinstance(questions, list) else []
        except:
            return []
            
    async def _generate_search_queries(self, query: str, num_queries: int = 3) -> List[ResearchQuery]:
        """Generate search queries for research"""
        prompt = f"""Given this research topic, generate {num_queries} search queries to explore different aspects.
        Each query should have a specific research goal. Return as JSON array with 'query' and 'research_goal' fields."""
        
        response = await self.llm_provider.call(
            system_prompt="You are a research assistant generating targeted search queries.",
            user_prompt=prompt,
            temperature=0.1
        )
        
        try:
            queries = json.loads(response["content"])
            return [ResearchQuery(**q) for q in queries]
        except:
            return []

    async def _deep_research(
        self,
        query: str,
        breadth: int,
        depth: int,
        concurrency: int,
        previous_learnings: List[str] = None
    ) -> ResearchResult:
        """Conduct deep research using Firecrawl"""
        
        learnings = previous_learnings or []
        visited_urls = []
        
        # Generate search queries
        search_queries = await self._generate_search_queries(query, num_queries=breadth)
        
        # Create semaphore for concurrent requests
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_query(research_query: ResearchQuery) -> Dict:
            async with semaphore:
                try:
                    # Search using Firecrawl
                    result = await self.firecrawl.search(research_query.query, timeout=15000, limit=5)
                    
                    # Extract URLs
                    urls = [item.get("url") for item in result["data"] if item.get("url")]
                    
                    # Process content to extract learnings
                    processed_result = await self._process_search_result(
                        query=research_query.query,
                        search_result=result
                    )
                    
                    return {
                        "learnings": processed_result["learnings"],
                        "urls": urls,
                        "follow_up_questions": processed_result["follow_up_questions"]
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing query {research_query.query}: {str(e)}")
                    return {"learnings": [], "urls": [], "follow_up_questions": []}
                    
        # Process all queries concurrently
        results = await asyncio.gather(*[process_query(q) for q in search_queries])
        
        # Combine results
        all_learnings = learnings + [l for r in results for l in r["learnings"]]
        all_urls = visited_urls + [u for r in results for u in r["urls"]]
        follow_up_questions = [q for r in results for q in r["follow_up_questions"]]
        
        # If we have more depth to go and follow-up questions
        if depth > 1 and follow_up_questions:
            next_query = "\n".join([
                f"Previous research goal: {query}",
                "Follow-up questions to explore:",
                "\n".join(f"- {q}" for q in follow_up_questions[:3])
            ])
            
            deeper_results = await self._deep_research(
                query=next_query,
                breadth=max(1, breadth // 2),
                depth=depth - 1,
                concurrency=concurrency,
                previous_learnings=all_learnings
            )
            
            all_learnings.extend(deeper_results["learnings"])
            all_urls.extend(deeper_results["visited_urls"])
            
        return {
            "learnings": list(set(all_learnings)),
            "visited_urls": list(set(all_urls)),
            "follow_up_questions": follow_up_questions
        }
