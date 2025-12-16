from mcp.server.fastmcp import FastMCP
import wikipedia

# Initialize the Server
mcp = FastMCP("WikiServer")

@mcp.tool()
def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for a query and returns a summary.
    """
    try:
        # Limit to 2 sentences for concise context
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()