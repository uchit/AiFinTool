"""
Agent Coordinator Module - Complete Financial Agent with Modular Architecture

This module provides the complete financial agent functionality with intelligent routing,
tool coordination, and backward compatibility. It replaces both modern_financial_agent.py
and financial_agent.py by providing all functionality in a single coordinated system.

Learning Objectives:
- Understand multi-tool coordination and intelligent routing
- Implement LLM-based decision making for tool selection
- Learn result synthesis from multiple data sources
- Build modular agent architecture
- Master PII protection in agent workflows

Your Task: Complete the missing implementations marked with YOUR CODE HERE

Key Features:
- Multi-tool coordination with intelligent routing
- Document analysis (10-K filings) for Apple, Google, Tesla
- Database queries with SQL auto-generation and PII protection
- Real-time market data from Yahoo Finance
- Complete backward compatibility for existing notebooks
- Modular architecture using helper modules
"""

import os
import logging
import re
import ast
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Complete Financial Agent with Dynamic Multi-Tool Coordination
    
    This class combines the functionality of the original modern_financial_agent.py
    and financial_agent.py into a single coordinated system using modular architecture.
    
    Architecture:
    - Document Tools (3): Individual SEC 10-K filing analysis for Apple, Google, Tesla
    - Function Tools (3): Database SQL queries, real-time market data, PII protection
    - Intelligent Routing: LLM-based tool selection and result synthesis
    - Backward Compatibility: Works with existing notebooks and code
    """
    
    def __init__(self, companies: List[str] = None, verbose: bool = False):
        """
        Initialize the complete financial agent with modular architecture.
        
        Args:
            companies: List of company symbols (default: ["AAPL", "GOOGL", "TSLA"])
            verbose: Whether to show detailed operation information
        """
        self.companies = companies if companies is not None else ["AAPL", "GOOGL", "TSLA"]
        self.verbose = verbose
        self.project_root = Path.cwd()  # Use current working directory
        
        # Company metadata
        self.company_info = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
            "TSLA": {"name": "Tesla Inc.", "sector": "Automotive"}
        }
        
        # Storage for tools and engines
        self.document_tools = []
        self.function_tools = []
        self.llm = None
    
        
        self._configure_settings()
        
        # Don't auto-initialize tools - create them lazily when first needed
        self._tools_initialized = False
        
        if self.verbose:
            print("✅ Financial Agent Coordinator Initialized")
            print(f"   Companies: {self.companies}")
            print(f"   Tools will be created automatically when first query is made")
    
  
    def _configure_settings(self):
        """Configure LlamaIndex settings with Vocareum API compatibility
        
        TODO: Set up the LLM and embedding model for intelligent routing
        
        Requirements:
        - Create OpenAI LLM with "gpt-3.5-turbo" model and temperature=0
        - Create OpenAIEmbedding with "text-embedding-ada-002" model
        - Use api_base parameter for Vocareum API compatibility (both models)
        - Set Settings.llm and Settings.embed_model
        - Store LLM reference in self.llm for routing decisions
        
        IMPORTANT NOTE FOR VOCAREUM:
        LlamaIndex requires the api_base parameter to work with Vocareum's OpenAI endpoint.
        Get the base URL from environment: os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
        Pass it as api_base parameter to both OpenAI() and OpenAIEmbedding() constructors.
        """
        api_base = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            os.environ["OPENAI_API_KEY"] = "DUMMY_KEY"
            api_key = "DUMMY_KEY"
        self._llm_available = api_key not in {"DUMMY_KEY", "dummy", "dummy_key"}
        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_base=api_base)
        Settings.llm = self.llm
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_base=api_base)

    def setup(self, document_tools: List = None, function_tools: List = None):
        """
        Setup all components using the modular architecture.
        
        Args:
            document_tools: Optional pre-created document tools
            function_tools: Optional pre-created function tools
            
        This method initializes all tools and sets up the routing system.
        If tools are not provided, they will be created automatically.
        """
        if self.verbose:
            print("🔧 Setting up Advanced Financial Agent (Modular Architecture)...")
        
        try:
            if document_tools is not None and function_tools is not None:
                # Use provided tools
                self.document_tools = document_tools
                self.function_tools = function_tools
            else:
                # Create tools automatically
                self._create_tools()
            
            if self.verbose:
                status = self.get_status()
                print(f"✅ Setup complete: {status['document_tools']} document tools, {status['function_tools']} function tools")
                print(f"🎯 System ready: {'✅' if status['ready'] else '❌'}")
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            if self.verbose:
                print(f"❌ Setup failed: {e}")
    
    def _create_tools(self):
        """Create all tools automatically using helper modules
        
        TODO: Import and use the DocumentToolsManager and FunctionToolsManager
        to create all necessary tools for the coordinator.
        
        Steps:
        1. Import DocumentToolsManager from .document_tools
        2. Import FunctionToolsManager from .function_tools
        3. Create instances and call their build methods
        4. Store results in self.document_tools and self.function_tools
        """
        from helper_modules.document_tools import DocumentToolsManager
        from helper_modules.function_tools import FunctionToolsManager

        doc_manager = DocumentToolsManager(companies=self.companies, verbose=self.verbose)
        self.document_tools = doc_manager.build_document_tools()

        func_manager = FunctionToolsManager(verbose=self.verbose)
        self.function_tools = func_manager.create_function_tools()

    def _check_and_apply_pii_protection(self, tool_name: str, result: str) -> str:
        """Check if database results need PII protection and apply it automatically
        
        This method automatically detects when database queries return sensitive information
        and applies appropriate PII protection using the PII protection tool from function_tools.
        
        Args:
            tool_name: Name of the tool that generated the result
            result: Raw result string from the tool
            
        Returns:
            Protected result string with PII masked if necessary
        """
        
        # Only apply to database query results
        if "database_query_tool" not in tool_name:
            return result
        
        # Check if result contains column information
        if "COLUMNS:" not in result:
            return result
        
        # Extract column names from result
        columns = []
        for line in result.splitlines():
            if line.strip().startswith("COLUMNS:"):
                col_text = line.split("COLUMNS:", 1)[1].strip()
                try:
                    parsed = ast.literal_eval(col_text)
                    if isinstance(parsed, list):
                        columns = parsed
                except Exception:
                    columns = [c.strip() for c in col_text.strip('[]').split(',') if c.strip()]
                break

        pii_fields = self._detect_pii_fields(columns)
        if not pii_fields:
            return result

        # Find PII protection tool and apply masking
        for tool in self.function_tools:
            if hasattr(tool, 'metadata') and tool.metadata.name == 'pii_protection_tool':
                try:
                    return tool.call(result, str(columns))
                except Exception:
                    return result

        return result  # PII tool not available
    
    def _detect_pii_fields(self, field_names: list) -> set:
        """Detect which fields contain PII based on field names
        
        This method identifies potentially sensitive database fields that need protection.
        
        Args:
            field_names: List of database column names
            
        Returns:
            Set of field names that contain PII
        """
        # Define PII field patterns
        patterns = [
            'name', 'first', 'last', 'email', 'phone', 'address',
            'ssn', 'social', 'dob', 'birth', 'account',
        ]
        pii_fields = set()
        for field in field_names:
            field_lower = str(field).lower()
            if any(p in field_lower for p in patterns):
                pii_fields.add(field)
        return pii_fields
    
    def _simple_routing(self, query: str) -> List[Any]:
        """Rule-based routing fallback when LLM routing is unavailable"""
        query_lower = query.lower()
        selected_tools = []

        def add_tool_by_name(name: str, tools: List):
            for tool in tools:
                if hasattr(tool, 'metadata') and tool.metadata.name == name:
                    if tool not in selected_tools:
                        selected_tools.append(tool)

        # Document tools by company
        if 'apple' in query_lower or 'aapl' in query_lower:
            add_tool_by_name('AAPL_10k_filing_tool', self.document_tools)
        if 'google' in query_lower or 'googl' in query_lower or 'alphabet' in query_lower:
            add_tool_by_name('GOOGL_10k_filing_tool', self.document_tools)
        if 'tesla' in query_lower or 'tsla' in query_lower:
            add_tool_by_name('TSLA_10k_filing_tool', self.document_tools)

        # Market data queries
        if any(k in query_lower for k in ['price', 'stock', 'market', 'share']):
            add_tool_by_name('finance_market_search_tool', self.function_tools)

        # Database queries
        if any(k in query_lower for k in ['customer', 'portfolio', 'holding', 'database', 'account']):
            add_tool_by_name('database_query_tool', self.function_tools)

        # If nothing matched, default to all document tools
        if not selected_tools:
            selected_tools.extend(self.document_tools)

        return selected_tools

    def _intelligent_routing(self, query: str) -> List[Any]:
        """LLM-based routing with fallback to simple rules"""
        if self.llm is None or not self._llm_available:
            return self._simple_routing(query)

        tools = self.document_tools + self.function_tools
        if not tools:
            return []

        tool_descriptions = []
        for idx, tool in enumerate(tools, start=1):
            name = tool.metadata.name if hasattr(tool, 'metadata') else f"tool_{idx}"
            desc = tool.metadata.description if hasattr(tool, 'metadata') else ''
            tool_descriptions.append(f"{idx}. {name}: {desc}")

        prompt = (
            "You are a routing assistant. Select the tool numbers needed to answer the query.\n"
            "Guidelines:\n"
            "- Customer/portfolio/account/holdings questions -> database_query_tool\n"
            "- Stock price/market/volume/market cap questions -> finance_market_search_tool\n"
            "- Company-specific SEC 10-K questions -> the matching company 10-K tool\n"
            "- If a query needs both company filings and customer data, select multiple tools\n"
            "Return a comma-separated list of tool numbers only.\n"
            f"\nQuery: {query}\n"
            f"\nTools:\n" + "\n".join(tool_descriptions)
        )

        try:
            response = self.llm.complete(prompt)
            text = response.text.strip()
            indices = [int(x) for x in re.findall(r"\d+", text)]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(tools):
                    selected.append(tools[idx - 1])
            return selected or self._simple_routing(query)
        except Exception:
            return self._simple_routing(query)

    def _synthesize_results(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Synthesize multiple tool results into a single response"""
        if self.llm is None or not self._llm_available:
            lines = [f"{item['tool']}: {item['result']}" for item in results]
            return "\n".join(lines)

        context = "\n".join([f"Tool {item['tool']}: {item['result']}" for item in results])
        prompt = (
            "You are a financial analyst. Provide a concise, well-structured answer that integrates\n"
            "all relevant tool outputs into a single coherent response. Resolve overlaps, cite which\n"
            "tool provided which facts, and clearly combine document analysis with database and market data\n"
            "when both are present.\n"
            f"\nQuestion: {question}\n"
            f"\nTool Outputs:\n{context}\n"
        )
        response = self.llm.complete(prompt)
        return response.text.strip()

    def _route_query(self, query: str) -> List[Tuple[str, str, Any]]:
        """Use LLM to intelligently route query to appropriate tools
        
        This method analyzes the user's query and determines which tools are needed
        to provide a complete answer, then executes those tools and returns results.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of tuples: (tool_name, tool_description, result)
        """
        
        # Build routing logic
        tools = self._intelligent_routing(query)
        if not tools:
            tools = self._simple_routing(query)

        results = []
        for tool in tools:
            tool_name = tool.metadata.name if hasattr(tool, 'metadata') else 'unknown_tool'
            tool_desc = tool.metadata.description if hasattr(tool, 'metadata') else ''
            try:
                if hasattr(tool, 'query_engine'):
                    tool_result = tool.query_engine.query(query)
                else:
                    tool_result = tool.call(query)
                result_str = str(tool_result)
                result_str = self._check_and_apply_pii_protection(tool_name, result_str)
                results.append((tool_name, tool_desc, result_str))
            except Exception as e:
                results.append((tool_name, tool_desc, f"Tool error: {e}"))

        return results
    
    def query(self, question: str, verbose: bool = None) -> str:
        """Process query with dynamic tool routing and result synthesis
        
        This is the main entry point for the financial agent. It handles:
        1. Tool routing and selection using LLM
        2. Multi-tool execution 
        3. Result synthesis for comprehensive answers
        4. Automatic PII protection
        
        Args:
            question: User's financial question
            verbose: Whether to show detailed processing info
            
        Returns:
            Comprehensive answer synthesized from relevant tools
        """
        
        # Use instance verbose if parameter not provided
        if verbose is None:
            verbose = self.verbose
        
        # Ensure tools are initialized
        if not self._tools_initialized:
            self.setup()
            self._tools_initialized = True
        
        if verbose:
            print(f"🎯 Query: {question}")
        
        # Implement query processing workflow
        results = self._route_query(question)
        if not results:
            return "No relevant tools available to answer the query."

        if verbose:
            for tool_name, _, _ in results:
                print(f"   🔧 Used tool: {tool_name}")

        if len(results) == 1:
            return results[0][2]

        synthesis_inputs = [
            {"tool": tool_name, "result": result}
            for tool_name, _, result in results
        ]
        return self._synthesize_results(question, synthesis_inputs)
    
    def list_available_tools(self) -> List[str]:
        """Return a flat list of available tool names"""
        tools = []
        for tool in self.document_tools + self.function_tools:
            if hasattr(tool, 'metadata'):
                tools.append(tool.metadata.name)
        return tools

    def get_available_tools(self) -> Dict[str, Any]:
        """
        Get information about available tools with full compatibility.
        
        Returns:
            Dictionary with comprehensive tool information
        """
        return {
            "document_tools": ["apple", "google", "tesla"] if len(self.document_tools) >= 3 else [],
            "function_tools": ["sql", "market", "pii"] if len(self.function_tools) >= 3 else [],
            "total_tools": len(self.document_tools) + len(self.function_tools),
            "document_tool_count": len(self.document_tools),
            "function_tool_count": len(self.function_tools)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status with full compatibility.
        
        Returns:
            Dictionary with detailed status information
        """
        tool_count = len(self.document_tools) + len(self.function_tools)
        system_ready = len(self.document_tools) >= 3 and len(self.function_tools) >= 3
        
        return {
            "companies": self.companies,
            "document_tools": len(self.document_tools),
            "function_tools": len(self.function_tools),
            "total_tools": tool_count,
            "ready": system_ready,
            "architecture": "modular",
            "coordinator_ready": system_ready,
            "available_companies": ['AAPL', 'GOOGL', 'TSLA'],
            "capabilities": [
                "Document analysis (10-K filings)",
                "Database queries (customer portfolios)",
                "Real-time market data",
                "PII protection",
                "Multi-tool coordination",
                "Intelligent routing"
            ],
            "system_ready": system_ready
        }
