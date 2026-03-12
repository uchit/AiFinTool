"""
Test Function Tools Module - Educational Testing Framework

This test file helps you verify your FunctionToolsManager implementation step by step.
Run this file to check your progress and get helpful debugging hints.

Usage: python tests/test_function_tools.py

The tests are designed to:
1. Check each function tool individually
2. Provide specific error messages and hints
3. Help you understand what went wrong and how to fix it
4. Build confidence as you progress through the implementation

Test Categories:
- Configuration Tests: Verify LLM setup and database connection
- Database Tool Tests: Check SQL generation and execution
- Market Data Tool Tests: Verify API integration and data fetching
- PII Protection Tool Tests: Check privacy protection functionality
- Integration Tests: End-to-end tool functionality verification
"""

import os
import sys
from pathlib import Path

# Ensure API key exists for tests (use dummy for local runs)
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'DUMMY_KEY'

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from helper_modules.function_tools import FunctionToolsManager
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.tools import FunctionTool
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Hint: Make sure you're running this from the starter_code directory")
    print("💡 Hint: Check that all required packages are installed: pip install -r requirements.txt")
    sys.exit(1)
class FunctionToolsTest:
    """Educational test framework for FunctionToolsManager"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.verbose = True
        
    def print_test_header(self, test_name: str):
        """Print a formatted test header"""
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {test_name}")
        print(f"{'='*60}")
    
    def print_success(self, message: str):
        """Print a success message"""
        print(f"✅ {message}")
        self.tests_passed += 1
    
    def print_failure(self, message: str, hint: str = None):
        """Print a failure message with optional hint"""
        print(f"❌ {message}")
        if hint:
            print(f"💡 Hint: {hint}")
        self.tests_failed += 1
    
    def print_summary(self):
        """Print test summary"""
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*60}")
        print(f"📊 Test Summary")
        print(f"{'='*60}")
        print(f"✅ Passed: {self.tests_passed}/{total}")
        print(f"❌ Failed: {self.tests_failed}/{total}")
        
        if self.tests_failed == 0:
            print(f"\n🎉 Congratulations! All tests passed!")
            print(f"🎯 Your FunctionToolsManager implementation is working correctly!")
            print(f"🚀 You're ready to move on to agent_coordinator.py")
        else:
            print(f"\n🔧 Keep working! Fix the failing tests and run again.")
            print(f"📚 Read the hints carefully - they'll guide you to the solution.")
    
    def test_environment_setup(self):
        """Test if environment is properly configured"""
        self.print_test_header("Environment Setup")
        
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            self.print_failure(
                "OpenAI API key not found",
                "Set OPENAI_API_KEY in your .env file or environment variables"
            )
            return False
        else:
            self.print_success("OpenAI API key configured")
        
        # Check database file
        db_path = project_root / "data" / "financial.db"
        if not db_path.exists():
            self.print_failure(
                "Database file not found",
                "Make sure financial.db exists in the data directory. Run build_database.py first."
            )
            return False
        else:
            self.print_success("Database file found")
        
        return True
    
    def test_initialization(self):
        """Test FunctionToolsManager initialization"""
        self.print_test_header("FunctionToolsManager Initialization")
        
        try:
            manager = FunctionToolsManager(verbose=False)
            self.print_success("FunctionToolsManager created successfully")
            
            # Check database path
            if hasattr(manager, 'db_path') and manager.db_path.exists():
                self.print_success("Database path properly set")
            else:
                self.print_failure(
                    "Database path not properly configured",
                    "Make sure self.db_path points to the financial.db file"
                )
            
            # Check schema
            if hasattr(manager, 'db_schema') and len(manager.db_schema) > 100:
                self.print_success("Database schema loaded")
            else:
                self.print_failure(
                    "Database schema not properly loaded",
                    "Check that _get_database_schema() returns detailed schema information"
                )
            
            # Check function_tools list
            if hasattr(manager, 'function_tools') and isinstance(manager.function_tools, list):
                self.print_success("Function tools list initialized")
            else:
                self.print_failure(
                    "Function tools list not properly initialized",
                    "Make sure self.function_tools = [] in __init__"
                )
            
            return manager
            
        except Exception as e:
            self.print_failure(
                f"Failed to create FunctionToolsManager: {e}",
                "Check your __init__ method implementation"
            )
            return None
    
    def test_configuration(self, manager):
        """Test LlamaIndex settings configuration"""
        self.print_test_header("LlamaIndex Configuration")
        
        if manager is None:
            self.print_failure("Cannot test configuration - manager not created", "Fix initialization first")
            return False
        
        # Check if LLM is configured
        if hasattr(manager, 'llm') and manager.llm is not None:
            if isinstance(manager.llm, OpenAI):
                self.print_success("OpenAI LLM configured correctly")
            else:
                self.print_failure(
                    f"LLM is not OpenAI type. Got: {type(manager.llm)}",
                    "Use self.llm = OpenAI(model='gpt-3.5-turbo', temperature=0)"
                )
        else:
            self.print_failure(
                "self.llm not configured",
                "Implement _configure_settings() method to set self.llm"
            )
        
        # Check Settings configuration
        if hasattr(Settings, 'llm') and Settings.llm is not None:
            self.print_success("Settings.llm configured")
        else:
            self.print_failure(
                "Settings.llm not configured",
                "Set Settings.llm = self.llm in _configure_settings()"
            )
        
        return True
    
    def test_create_function_tools(self, manager):
        """Test function tools creation"""
        self.print_test_header("Creating Function Tools")
        
        if manager is None:
            self.print_failure("Cannot test tool creation - manager not created", "Fix initialization first")
            return False
        
        try:
            # Create the tools
            tools = manager.create_function_tools()
            
            # Check if tools were created
            if tools is None:
                self.print_failure(
                    "create_function_tools() returned None",
                    "Make sure to return self.function_tools at the end of the method"
                )
                return False
            
            if not isinstance(tools, list):
                self.print_failure(
                    f"create_function_tools() should return a list, got: {type(tools)}",
                    "Return self.function_tools which should be a list"
                )
                return False
            
            # Check number of tools
            expected_count = 3  # database, market, pii
            if len(tools) == expected_count:
                self.print_success(f"Created {expected_count} function tools")
            else:
                self.print_failure(
                    f"Expected {expected_count} tools, got {len(tools)}",
                    "Make sure all 3 tools are created: database_query_tool, finance_market_search_tool, pii_protection_tool"
                )
            
            # Check tool types
            for i, tool in enumerate(tools):
                if isinstance(tool, FunctionTool):
                    self.print_success(f"Tool {i+1} is correct FunctionTool type")
                else:
                    self.print_failure(
                        f"Tool {i+1} is not a FunctionTool, got: {type(tool)}",
                        "Use FunctionTool.from_defaults() to create tools"
                    )
            
            # Check tool names
            expected_names = ["database_query_tool", "finance_market_search_tool", "pii_protection_tool"]
            actual_names = [tool.metadata.name for tool in tools if hasattr(tool, 'metadata')]
            
            for expected_name in expected_names:
                if expected_name in actual_names:
                    self.print_success(f"Found tool: {expected_name}")
                else:
                    self.print_failure(
                        f"Missing tool: {expected_name}",
                        f"Make sure to create FunctionTool with name='{expected_name}'"
                    )
            
            # Check tool descriptions
            for tool in tools:
                if hasattr(tool, 'metadata') and hasattr(tool.metadata, 'description'):
                    if len(tool.metadata.description) > 30:  # Reasonable description length
                        self.print_success(f"Tool {tool.metadata.name} has proper description")
                    else:
                        self.print_failure(
                            f"Tool {tool.metadata.name} has insufficient description",
                            "Provide a detailed description explaining what the tool does"
                        )
                else:
                    self.print_failure(
                        f"Tool missing metadata or description",
                        "Make sure FunctionTool.from_defaults() includes name and description"
                    )
            
            return tools
            
        except Exception as e:
            self.print_failure(
                f"Error creating function tools: {e}",
                "Check your create_function_tools() implementation. Make sure all YOUR CODE HERE sections are completed."
            )
            return None
    
    def test_database_tool_basic(self, manager, tools):
        """Test basic database tool functionality"""
        self.print_test_header("Database Tool Basic Functionality")
        
        if not tools or len(tools) == 0:
            self.print_failure("No tools available for testing", "Fix tool creation first")
            return False
        
        # Find database tool
        db_tool = None
        for tool in tools:
            if hasattr(tool, 'metadata') and 'database' in tool.metadata.name:
                db_tool = tool
                break
        
        if db_tool is None:
            self.print_failure(
                "Database tool not found",
                "Make sure database_query_tool is created with correct name"
            )
            return False
        
        try:
            # Test simple query (this will likely fail until implemented)
            result = db_tool.call("Show me all customers")
            
            if "not implemented" in result.lower():
                self.print_failure(
                    "Database tool not implemented yet",
                    "Complete the database_query_tool function implementation"
                )
            elif "error" in result.lower():
                self.print_failure(
                    f"Database tool returned error: {result[:100]}...",
                    "Check your SQL generation and database connection logic"
                )
            else:
                self.print_success("Database tool executed without errors")
                
        except Exception as e:
            self.print_failure(
                f"Error calling database tool: {e}",
                "Check that the database_query_tool function is properly implemented"
            )
        
        return True
    
    def test_market_tool_basic(self, manager, tools):
        """Test basic market data tool functionality"""
        self.print_test_header("Market Data Tool Basic Functionality")
        
        if not tools or len(tools) == 0:
            self.print_failure("No tools available for testing", "Fix tool creation first")
            return False
        
        # Find market tool
        market_tool = None
        for tool in tools:
            if hasattr(tool, 'metadata') and 'market' in tool.metadata.name:
                market_tool = tool
                break
        
        if market_tool is None:
            self.print_failure(
                "Market tool not found",
                "Make sure finance_market_search_tool is created with correct name"
            )
            return False
        
        try:
            # Test market query
            result = market_tool.call("What is Apple's stock price?")
            
            if "not implemented" in result.lower():
                self.print_failure(
                    "Market tool not implemented yet",
                    "Complete the finance_market_search_tool function implementation"
                )
            elif "error" in result.lower() and "not implemented" not in result.lower():
                self.print_failure(
                    f"Market tool returned error: {result[:100]}...",
                    "Check your market data fetching logic and API integration"
                )
            else:
                self.print_success("Market tool executed without errors")
                
        except Exception as e:
            self.print_failure(
                f"Error calling market tool: {e}",
                "Check that the finance_market_search_tool function is properly implemented"
            )
        
        return True
    
    def test_pii_tool_basic(self, manager, tools):
        """Test basic PII protection tool functionality"""
        self.print_test_header("PII Protection Tool Basic Functionality")
        
        if not tools or len(tools) == 0:
            self.print_failure("No tools available for testing", "Fix tool creation first")
            return False
        
        # Find PII tool
        pii_tool = None
        for tool in tools:
            if hasattr(tool, 'metadata') and 'pii' in tool.metadata.name:
                pii_tool = tool
                break
        
        if pii_tool is None:
            self.print_failure(
                "PII tool not found",
                "Make sure pii_protection_tool is created with correct name"
            )
            return False
        
        try:
            # Test PII protection
            test_data = "Customer: John Doe, Email: john@example.com"
            result = pii_tool.call(test_data, "['name', 'email']")
            
            if result == test_data:
                self.print_failure(
                    "PII tool returned unchanged data",
                    "Implement PII masking logic in pii_protection_tool function"
                )
            else:
                self.print_success("PII tool modified the input data (masking applied)")
                
        except Exception as e:
            self.print_failure(
                f"Error calling PII tool: {e}",
                "Check that the pii_protection_tool function is properly implemented"
            )
        
        return True
    
    def test_get_tools_method(self, manager):
        """Test the get_tools method"""
        self.print_test_header("Get Tools Method")
        
        if manager is None:
            self.print_failure("Cannot test get_tools - manager not created", "Fix initialization first")
            return False
        
        try:
            tools = manager.get_tools()
            if isinstance(tools, list):
                self.print_success("get_tools() returns a list")
            else:
                self.print_failure(
                    f"get_tools() should return a list, got: {type(tools)}",
                    "Return self.function_tools from get_tools() method"
                )
        except Exception as e:
            self.print_failure(
                f"Error calling get_tools(): {e}",
                "Make sure get_tools() method is implemented correctly"
            )
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting FunctionToolsManager Testing Framework")
        print("📋 This will test your implementation step by step")
        
        # Environment setup
        if not self.test_environment_setup():
            print("\n⚠️  Environment issues detected. Please fix before continuing.")
            self.print_summary()
            return
        
        # Initialization
        manager = self.test_initialization()
        
        # Configuration
        self.test_configuration(manager)
        
        # Tool creation
        tools = self.test_create_function_tools(manager)
        
        # Basic tool functionality tests
        self.test_database_tool_basic(manager, tools)
        self.test_market_tool_basic(manager, tools)
        self.test_pii_tool_basic(manager, tools)
        
        # Helper methods
        self.test_get_tools_method(manager)
        
        # Summary
        self.print_summary()

def main():
    """Run the educational test framework"""
    tester = FunctionToolsTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main()


import pytest
import os
import sqlite3
from pathlib import Path
import sys

# Add helper_modules to path
sys.path.append(str(Path(__file__).parent.parent / "helper_modules"))

class TestFunctionTools:
    """Test suite for function tools implementation"""
    
    def test_environment_setup(self):
        """Test 1: Verify environment and required packages"""
        print("\n" + "="*60)
        print("TEST 1: Environment Setup")
        print("="*60)
        
        # Check required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing environment variables: {missing_vars}")
            print("💡 HINT: Create a .env file with your OpenAI API key")
            print("   Example: OPENAI_API_KEY=your_key_here")
            pytest.fail(f"Missing environment variables: {missing_vars}")
        
        # Check required packages
        required_packages = [
            'yfinance', 'sqlite3', 'pathlib', 'dotenv',
            'llama_index.core', 'llama_index.llms.openai', 'llama_index.embeddings.openai'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                if package == 'sqlite3':
                    import sqlite3
                elif package == 'pathlib':
                    from pathlib import Path
                elif package == 'dotenv':
                    from dotenv import load_dotenv
                elif package == 'yfinance':
                    import yfinance as yf
                elif 'llama_index' in package:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {missing_packages}")
            print("💡 HINT: Install missing packages with:")
            print("   pip install -r requirements.txt")
            pytest.fail(f"Missing packages: {missing_packages}")
        
        print("✅ All environment requirements satisfied")
    
    def test_imports(self):
        """Test 2: Check that function_tools module imports correctly"""
        print("\n" + "="*60)
        print("TEST 2: Module Imports")
        print("="*60)
        
        try:
            from function_tools import FunctionToolsManager
            print("✅ FunctionToolsManager imported successfully")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 HINT: Check that function_tools.py is in helper_modules/")
            print("💡 HINT: Verify all imports in function_tools.py are correct")
            pytest.fail(f"Import error: {e}")
        
        # Test LlamaIndex imports
        try:
            from llama_index.core.tools import FunctionTool
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding
            print("✅ LlamaIndex components imported successfully")
        except ImportError as e:
            print(f"❌ LlamaIndex import error: {e}")
            print("💡 HINT: Install llama-index with:")
            print("   pip install llama-index llama-index-llms-openai llama-index-embeddings-openai")
            pytest.fail(f"LlamaIndex import error: {e}")
    
    def test_initialization(self):
        """Test 3: Test FunctionToolsManager initialization"""
        print("\n" + "="*60)
        print("TEST 3: Manager Initialization")
        print("="*60)
        
        try:
            from function_tools import FunctionToolsManager
            
            # Test initialization
            manager = FunctionToolsManager(verbose=True)
            
            # Check attributes
            assert hasattr(manager, 'verbose'), "Manager missing 'verbose' attribute"
            assert hasattr(manager, 'project_root'), "Manager missing 'project_root' attribute"
            assert hasattr(manager, 'db_path'), "Manager missing 'db_path' attribute"
            assert hasattr(manager, 'db_schema'), "Manager missing 'db_schema' attribute"
            assert hasattr(manager, 'function_tools'), "Manager missing 'function_tools' attribute"
            assert hasattr(manager, 'llm'), "Manager missing 'llm' attribute"
            
            print("✅ FunctionToolsManager initialized with all required attributes")
            
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            print("💡 HINT: Check your __init__ method implementation")
            print("💡 HINT: Verify _configure_settings() method is working")
            pytest.fail(f"Initialization error: {e}")
    
    def test_schema_retrieval(self):
        """Test 4: Test database schema retrieval"""
        print("\n" + "="*60)
        print("TEST 4: Database Schema Retrieval")
        print("="*60)
        
        try:
            from function_tools import FunctionToolsManager
            manager = FunctionToolsManager(verbose=False)
            
            # Check schema content
            schema = manager.db_schema
            assert isinstance(schema, str), "Schema should be a string"
            assert len(schema) > 0, "Schema should not be empty"
            
            # Check for key table names
            expected_tables = ['customers', 'portfolio_holdings', 'companies', 'market_data']
            for table in expected_tables:
                if table not in schema.lower():
                    print(f"⚠️  Warning: Table '{table}' not found in schema")
            
            print("✅ Database schema retrieved successfully")
            print(f"📋 Schema length: {len(schema)} characters")
            
        except Exception as e:
            print(f"❌ Schema retrieval error: {e}")
            print("💡 HINT: Check your _get_database_schema() method")
            print("💡 HINT: Ensure database file exists at correct path")
            print("💡 HINT: Verify database connection logic")
            pytest.fail(f"Schema retrieval error: {e}")
    
    def test_tool_creation(self):
        """Test 5: Test FunctionTool creation"""
        print("\n" + "="*60)
        print("TEST 5: Function Tool Creation")
        print("="*60)
        
        try:
            from function_tools import FunctionToolsManager
            manager = FunctionToolsManager(verbose=False)
            
            # Create tools
            tools = manager.create_function_tools()
            
            # Check tools were created
            assert isinstance(tools, list), "create_function_tools should return a list"
            
            if len(tools) == 0:
                print("⚠️  No tools created yet - this is expected if not implemented")
                print("💡 HINT: Implement create_function_tools() method")
                print("💡 HINT: Create FunctionTool objects for:")
                print("   - database_query_tool")
                print("   - finance_market_search_tool") 
                print("   - pii_protection_tool")
                return
            
            # Check tool properties
            from llama_index.core.tools import FunctionTool
            for i, tool in enumerate(tools):
                assert isinstance(tool, FunctionTool), f"Tool {i} should be a FunctionTool instance"
                assert hasattr(tool, 'metadata'), f"Tool {i} missing metadata"
                assert hasattr(tool.metadata, 'name'), f"Tool {i} missing name"
            
            print(f"✅ Created {len(tools)} function tools successfully")
            for tool in tools:
                print(f"   📋 Tool: {tool.metadata.name}")
            
        except Exception as e:
            print(f"❌ Tool creation error: {e}")
            print("💡 HINT: Implement create_function_tools() method")
            print("💡 HINT: Use FunctionTool.from_defaults() to create tools")
            print("💡 HINT: Make sure method signatures match function definitions")
            pytest.fail(f"Tool creation error: {e}")
    
    def test_database_functionality(self):
        """Test 6: Test database query functionality"""
        print("\n" + "="*60)
        print("TEST 6: Database Query Functionality")
        print("="*60)
        
        try:
            from function_tools import FunctionToolsManager
            manager = FunctionToolsManager(verbose=False)
            
            # Test multiple query patterns (aggregation, join, filtering)
            test_queries = [
                "How many customers do we have?",
                "Show customers who own Tesla stock and their share counts.",
                "Total current_value by symbol for all holdings.",
            ]

            for test_query in test_queries:
                result = manager.database_query_tool(test_query)
                assert isinstance(result, str), "Database query should return a string"

                if "not implemented" in result.lower():
                    print("⚠️  Database query tool not implemented yet")
                    print("💡 HINT: Complete the database_query_tool() method")
                    print("💡 HINT: Use LLM to generate SQL from natural language")
                    print("💡 HINT: Execute SQL against SQLite database")
                    print("💡 HINT: Format results as string with column information")
                    return

                if "error" in result.lower():
                    pytest.fail(f"Database query error for '{test_query}': {result[:120]}...")

                if "COLUMNS:" not in result:
                    pytest.fail("Database result missing column metadata (COLUMNS:)")

                print(f"✅ Database query executed successfully")
                print(f"📋 Query: {test_query}")
                print(f"📋 Result preview: {result[:100]}...")
            
        except Exception as e:
            print(f"❌ Database query error: {e}")
            print("💡 HINT: Check database_query_tool() implementation")
            print("💡 HINT: Verify database connection and SQL execution")
            print("💡 HINT: Handle errors gracefully")
            pytest.fail(f"Database query error: {e}")
    
    def test_market_data_functionality(self):
        """Test 7: Test market data functionality"""
        print("\n" + "="*60)
        print("TEST 7: Market Data Functionality")
        print("="*60)
        
        try:
            from function_tools import FunctionToolsManager
            manager = FunctionToolsManager(verbose=False)
            
            # Test market data method
            test_query = "Get AAPL stock price"
            result = manager.finance_market_search_tool(test_query)
            
            assert isinstance(result, str), "Market search should return a string"
            
            if "not implemented" in result.lower():
                print("⚠️  Market search tool not implemented yet")
                print("💡 HINT: Complete the finance_market_search_tool() method")
                print("💡 HINT: Use yfinance to get real-time market data")
                print("💡 HINT: Parse query to extract stock symbols")
                print("💡 HINT: Format results with price and volume information")
                return
            
            print(f"✅ Market data retrieved successfully")
            print(f"📋 Query: {test_query}")
            print(f"📋 Result preview: {result[:100]}...")
            
        except Exception as e:
            print(f"❌ Market data error: {e}")
            print("💡 HINT: Check finance_market_search_tool() implementation")
            print("💡 HINT: Verify yfinance is installed and working")
            print("💡 HINT: Handle API errors gracefully")
            pytest.fail(f"Market data error: {e}")
    
    def test_pii_protection(self):
        """Test 8: Test PII protection functionality"""
        print("\n" + "="*60)
        print("TEST 8: PII Protection Functionality")
        print("="*60)
        
        try:
            from function_tools import FunctionToolsManager
            manager = FunctionToolsManager(verbose=False)
            
            # Test PII protection method
            test_data = "Customer: John Doe, Email: john.doe@example.com, Phone: 555-1234"
            result = manager.pii_protection_tool(test_data)
            
            assert isinstance(result, str), "PII protection should return a string"
            
            if "PII protection applied" in result and test_data in result:
                print("⚠️  PII protection tool not fully implemented yet")
                print("💡 HINT: Complete the pii_protection_tool() method")
                print("💡 HINT: Identify PII patterns (email, phone, names)")
                print("💡 HINT: Apply appropriate masking (e.g., jo***@example.com)")
                print("💡 HINT: Use column information to identify PII fields")
                return
            
            print(f"✅ PII protection applied successfully")
            print(f"📋 Original: {test_data}")
            print(f"📋 Protected: {result}")
            
        except Exception as e:
            print(f"❌ PII protection error: {e}")
            print("💡 HINT: Check pii_protection_tool() implementation")
            print("💡 HINT: Implement pattern matching for PII fields")
            print("💡 HINT: Apply consistent masking strategies")
            pytest.fail(f"PII protection error: {e}")

if __name__ == "__main__":
    # Run tests individually for better feedback
    test_suite = TestFunctionTools()
    
    tests = [
        ("Environment Setup", test_suite.test_environment_setup),
        ("Module Imports", test_suite.test_imports),
        ("Manager Initialization", test_suite.test_initialization),
        ("Schema Retrieval", test_suite.test_schema_retrieval),
        ("Tool Creation", test_suite.test_tool_creation),
        ("Database Functionality", test_suite.test_database_functionality),
        ("Market Data Functionality", test_suite.test_market_data_functionality),
        ("PII Protection", test_suite.test_pii_protection)
    ]
    
    print("Function Tools Test Suite")
    print("=" * 60)
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
        print()
