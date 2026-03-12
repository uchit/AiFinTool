"""
Test Document Tools Module - Educational Testing Framework

This test file helps you verify your DocumentToolsManager implementation step by step.
Run this file to check your progress and get helpful debugging hints.

Usage: python tests/test_document_tools.py

The tests are designed to:
1. Check each method individually
2. Provide specific error messages and hints
3. Help you understand what went wrong and how to fix it
4. Build confidence as you progress through the implementation

Test Categories:
- Configuration Tests: Verify LlamaIndex settings
- Initialization Tests: Check object creation and setup
- Document Processing Tests: Verify PDF loading and indexing
- Tool Creation Tests: Check QueryEngineTool creation
- Integration Tests: End-to-end functionality verification
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
    from helper_modules.document_tools import DocumentToolsManager
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.tools import QueryEngineTool
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Hint: Make sure you're running this from the starter_code directory")
    print("💡 Hint: Check that all required packages are installed: pip install -r requirements.txt")
    sys.exit(1)

class DocumentToolsTest:
    """Educational test framework for DocumentToolsManager"""
    
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
            print(f"🎯 Your DocumentToolsManager implementation is working correctly!")
            print(f"🚀 You're ready to move on to function_tools.py")
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
        
        # Check data directory
        data_dir = project_root / "data"
        if not data_dir.exists():
            self.print_failure(
                "Data directory not found",
                "Make sure the 'data' directory exists in your project root"
            )
            return False
        else:
            self.print_success("Data directory found")
        
        # Check PDF files
        pdf_files = ["AAPL_10K_2024.pdf", "GOOGL_10K_2024.pdf", "TSLA_10K_2024.pdf"]
        documents_dir = data_dir / "10k_documents"
        
        if not documents_dir.exists():
            self.print_failure(
                "Documents directory not found",
                "Make sure 'data/10k_documents' directory exists"
            )
            return False
        
        for pdf in pdf_files:
            pdf_path = documents_dir / pdf
            if not pdf_path.exists():
                self.print_failure(
                    f"PDF file missing: {pdf}",
                    f"Make sure {pdf} exists in data/10k_documents/"
                )
                return False
        
        self.print_success("All required PDF files found")
        return True
    
    def test_initialization(self):
        """Test DocumentToolsManager initialization"""
        self.print_test_header("DocumentToolsManager Initialization")
        
        try:
            manager = DocumentToolsManager(verbose=False)
            self.print_success("DocumentToolsManager created successfully")
            
            # Check default companies
            expected_companies = ["AAPL", "GOOGL", "TSLA"]
            if manager.companies == expected_companies:
                self.print_success("Default companies set correctly")
            else:
                self.print_failure(
                    f"Default companies incorrect. Expected: {expected_companies}, Got: {manager.companies}",
                    "Check the default value in __init__ method"
                )
            
            # Check company_info dictionary
            if hasattr(manager, 'company_info') and len(manager.company_info) == 3:
                self.print_success("Company info dictionary properly initialized")
            else:
                self.print_failure(
                    "Company info dictionary missing or incomplete",
                    "Make sure company_info is a dictionary with AAPL, GOOGL, TSLA keys"
                )
            
            # Check document_tools list
            if hasattr(manager, 'document_tools') and isinstance(manager.document_tools, list):
                self.print_success("Document tools list initialized")
            else:
                self.print_failure(
                    "Document tools list not properly initialized",
                    "Make sure self.document_tools = [] in __init__"
                )
            
            return manager
            
        except Exception as e:
            self.print_failure(
                f"Failed to create DocumentToolsManager: {e}",
                "Check your __init__ method implementation"
            )
            return None
    
    def test_configuration(self, manager):
        """Test LlamaIndex settings configuration"""
        self.print_test_header("LlamaIndex Configuration")
        
        if manager is None:
            self.print_failure("Cannot test configuration - manager not created", "Fix initialization first")
            return False
        
        # Check if Settings.llm is configured
        if hasattr(Settings, 'llm') and Settings.llm is not None:
            if isinstance(Settings.llm, OpenAI):
                self.print_success("OpenAI LLM configured correctly")
            else:
                self.print_failure(
                    f"LLM is not OpenAI type. Got: {type(Settings.llm)}",
                    "Use Settings.llm = OpenAI(model='gpt-3.5-turbo', temperature=0)"
                )
        else:
            self.print_failure(
                "Settings.llm not configured",
                "Implement _configure_settings() method to set Settings.llm"
            )
        
        # Check if Settings.embed_model is configured
        if hasattr(Settings, 'embed_model') and Settings.embed_model is not None:
            if isinstance(Settings.embed_model, OpenAIEmbedding):
                self.print_success("OpenAI embeddings configured correctly")
            else:
                self.print_failure(
                    f"Embedding model is not OpenAI type. Got: {type(Settings.embed_model)}",
                    "Use Settings.embed_model = OpenAIEmbedding(model='text-embedding-ada-002')"
                )
        else:
            self.print_failure(
                "Settings.embed_model not configured",
                "Implement _configure_settings() method to set Settings.embed_model"
            )
        
        return True
    
    def test_build_document_tools(self, manager):
        """Test document tools building process"""
        self.print_test_header("Building Document Tools")
        
        if manager is None:
            self.print_failure("Cannot test tool building - manager not created", "Fix initialization first")
            return False
        
        try:
            # Build the tools
            tools = manager.build_document_tools()
            
            # Check if tools were created
            if tools is None:
                self.print_failure(
                    "build_document_tools() returned None",
                    "Make sure to return self.document_tools at the end of the method"
                )
                return False
            
            if not isinstance(tools, list):
                self.print_failure(
                    f"build_document_tools() should return a list, got: {type(tools)}",
                    "Return self.document_tools which should be a list"
                )
                return False
            
            # Check number of tools
            expected_count = 3  # AAPL, GOOGL, TSLA
            if len(tools) == expected_count:
                self.print_success(f"Created {expected_count} document tools")
            else:
                self.print_failure(
                    f"Expected {expected_count} tools, got {len(tools)}",
                    "Make sure all 3 PDF files are processed and tools are added to the list"
                )
            
            # Check tool types
            for i, tool in enumerate(tools):
                if isinstance(tool, QueryEngineTool):
                    self.print_success(f"Tool {i+1} is correct QueryEngineTool type")
                else:
                    self.print_failure(
                        f"Tool {i+1} is not a QueryEngineTool, got: {type(tool)}",
                        "Use QueryEngineTool.from_defaults() to create tools"
                    )
            
            # Check tool names
            expected_names = ["AAPL_10k_filing_tool", "GOOGL_10k_filing_tool", "TSLA_10k_filing_tool"]
            actual_names = [tool.metadata.name for tool in tools]
            
            for expected_name in expected_names:
                if expected_name in actual_names:
                    self.print_success(f"Found tool: {expected_name}")
                else:
                    self.print_failure(
                        f"Missing tool: {expected_name}",
                        f"Check that tool names follow the pattern: 'COMPANY_10k_filing_tool'"
                    )
            
            # Check tool metadata
            for tool in tools:
                if hasattr(tool, 'metadata') and hasattr(tool.metadata, 'description'):
                    if len(tool.metadata.description) > 50:  # Reasonable description length
                        self.print_success(f"Tool {tool.metadata.name} has proper description")
                    else:
                        self.print_failure(
                            f"Tool {tool.metadata.name} has insufficient description",
                            "Provide a detailed description explaining what the tool does"
                        )
                else:
                    self.print_failure(
                        f"Tool missing metadata or description",
                        "Make sure QueryEngineTool.from_defaults() includes name and description"
                    )
            
            return tools
            
        except Exception as e:
            self.print_failure(
                f"Error building document tools: {e}",
                "Check your build_document_tools() implementation. Make sure all YOUR CODE HERE sections are completed."
            )
            return None
    
    def test_tool_functionality(self, manager, tools):
        """Test that tools can actually answer questions"""
        self.print_test_header("Tool Functionality")
        
        if not tools or len(tools) == 0:
            self.print_failure("No tools available for testing", "Fix tool building first")
            return False
        
        # Test querying a tool
        test_questions = [
            "What is the company's main business?",
            "What are the key risk factors?",
            "What was the revenue last year?"
        ]
        
        for i, tool in enumerate(tools[:1]):  # Test just the first tool to save time
            try:
                question = test_questions[0]
                result = tool.query_engine.query(question)
                
                if result and len(str(result)) > 10:  # Non-empty meaningful response
                    self.print_success(f"Tool {tool.metadata.name} successfully answered query")
                else:
                    self.print_failure(
                        f"Tool {tool.metadata.name} returned empty or very short response",
                        "Check that your vector index and query engine are properly configured"
                    )
                    
            except Exception as e:
                self.print_failure(
                    f"Error querying tool {tool.metadata.name}: {e}",
                    "Check that query_engine is properly created from the vector index"
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
                    "Return self.document_tools from get_tools() method"
                )
        except Exception as e:
            self.print_failure(
                f"Error calling get_tools(): {e}",
                "Make sure get_tools() method is implemented correctly"
            )
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting DocumentToolsManager Testing Framework")
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
        
        # Tool building
        tools = self.test_build_document_tools(manager)
        
        # Tool functionality
        self.test_tool_functionality(manager, tools)
        
        # Helper methods
        self.test_get_tools_method(manager)
        
        # Summary
        self.print_summary()

def main():
    """Run the educational test framework"""
    tester = DocumentToolsTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
