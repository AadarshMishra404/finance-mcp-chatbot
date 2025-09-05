import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import openai
from abc import ABC, abstractmethod

# Configuration
st.set_page_config(
    page_title="Finance News MCP Chatbot",
    page_icon="📈",
    layout="wide"
)

@dataclass
class ToolResult:
    """Standardized tool result format"""
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str = ""

class BaseTool(ABC):
    """Abstract base class for all MCP tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict:
        pass

class FinanceNewsTool(BaseTool):
    """Tool for fetching finance news using NewsAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    @property
    def name(self) -> str:
        return "finance_news"
    
    @property
    def description(self) -> str:
        return "Fetches latest finance and stock market news with summaries"
    
    def get_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for news (e.g., 'Tesla stock', 'market update')"
                        },
                        "category": {
                            "type": "string",
                            "description": "News category",
                            "enum": ["general", "business", "technology"]
                        },
                        "country": {
                            "type": "string",
                            "description": "Country code (e.g., 'us', 'in')",
                            "default": "us"
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "Number of articles to fetch (1-20)",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5
                        }
                    },
                    "required": []
                }
            }
        }
    
    def execute(self, query: str = "", category: str = "business", 
                country: str = "us", page_size: int = 5) -> ToolResult:
        try:
            # If no specific query, get general business/finance news
            if not query:
                endpoint = f"{self.base_url}/top-headlines"
                params = {
                    "apiKey": self.api_key,
                    "category": category,
                    "country": country,
                    "pageSize": page_size
                }
            else:
                # Search for specific finance-related news
                endpoint = f"{self.base_url}/everything"
                params = {
                    "apiKey": self.api_key,
                    "q": f"{query} AND (stock OR market OR finance OR business)",
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": page_size
                }
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "ok":
                return ToolResult(
                    success=False,
                    data=None,
                    error=data.get("message", "Unknown error from NewsAPI"),
                    tool_name=self.name
                )
            
            articles = data.get("articles", [])
            
            # Process articles for better display
            processed_articles = []
            for article in articles:
                processed_article = {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", ""),
                    "url_to_image": article.get("urlToImage", "")
                }
                processed_articles.append(processed_article)
            
            return ToolResult(
                success=True,
                data={
                    "articles": processed_articles,
                    "total_results": data.get("totalResults", 0),
                    "query": query,
                    "category": category
                },
                tool_name=self.name
            )
            
        except requests.RequestException as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Network error: {str(e)}",
                tool_name=self.name
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {str(e)}",
                tool_name=self.name
            )

class MCPOrchestrator:
    """Main orchestrator for MCP tools"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.tools: Dict[str, BaseTool] = {}
        self.conversation_history = []
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool with the orchestrator"""
        self.tools[tool.name] = tool
    
    def get_available_tools_schema(self) -> List[Dict]:
        """Get OpenAI function calling schema for all registered tools"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name
            )
        
        return self.tools[tool_name].execute(**kwargs)
    
    def chat(self, user_message: str) -> str:
        """Main chat function with intelligent tool orchestration"""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # System message for the assistant
        system_message = {
            "role": "system",
            "content": """You are a finance news assistant with access to real-time news data. 
            You help users get the latest finance and stock market news, and provide summaries and insights.
            
            When users ask about:
            - Latest news, market updates, or general finance news: Use the finance_news tool without specific query
            - Specific stocks, companies, or topics: Use the finance_news tool with the relevant query
            - Analysis or summaries: First get the news, then provide your analysis
            
            Always provide helpful summaries and context for the news you fetch. Be conversational and informative."""
        }
        
        messages = [system_message] + self.conversation_history[-10:]  # Keep last 10 messages for context
        
        try:
            # First, determine if we need to use tools
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.get_available_tools_schema(),
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                # Execute the requested tools
                tool_results = []
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    result = self.execute_tool(tool_name, **tool_args)
                    tool_results.append((tool_call.id, result))
                    
                    # Add tool call to conversation
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call.model_dump()]
                    })
                    
                    # Add tool result to conversation
                    if result.success:
                        tool_content = json.dumps(result.data)
                    else:
                        tool_content = f"Error: {result.error}"
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "content": tool_content,
                        "tool_call_id": tool_call.id
                    })
                
                # Get final response with tool results
                final_response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system_message] + self.conversation_history[-15:]
                )
                
                final_content = final_response.choices[0].message.content
                self.conversation_history.append({"role": "assistant", "content": final_content})
                
                return final_content
            
            else:
                # No tools needed, return direct response
                content = assistant_message.content
                self.conversation_history.append({"role": "assistant", "content": content})
                return content
                
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

def main():
    st.title("📈 Finance News MCP Chatbot")
    st.markdown("Get the latest finance news and market updates with AI-powered insights!")
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        newsapi_key = st.text_input(
            "NewsAPI Key",
            type="password",
            help="Get your free key from newsapi.org"
        )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            if "orchestrator" in st.session_state:
                st.session_state.orchestrator.clear_history()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🛠️ Available Tools")
        st.markdown("- **Finance News**: Latest market updates and news")
        st.markdown("- **More tools coming soon!**")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize orchestrator when API keys are provided
    if openai_key and newsapi_key:
        if "orchestrator" not in st.session_state:
            st.session_state.orchestrator = MCPOrchestrator(openai_key)
            
            # Register tools
            finance_tool = FinanceNewsTool(newsapi_key)
            st.session_state.orchestrator.register_tool(finance_tool)
            
            st.success("✅ MCP system initialized with Finance News tool!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about finance news, market updates, or specific stocks..."):
        if not (openai_key and newsapi_key):
            st.error("Please provide both OpenAI and NewsAPI keys in the sidebar.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Getting latest finance news..."):
                try:
                    response = st.session_state.orchestrator.chat(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    # Example usage and testing
    st.markdown("### 💡 Example Queries")
    st.markdown("""
    - "What's the latest market news?"
    - "Show me news about Tesla stock"
    - "Any updates on the stock market today?"
    - "Get me business news from India"
    """)
    
    main()