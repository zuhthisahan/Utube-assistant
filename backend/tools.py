import os
import sys
import io
import uuid
import re
import datetime
import asyncio
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import List

# Use non-interactive backend for server plotting (Prevents popup windows)
matplotlib.use('Agg')

from langchain_core.tools import Tool
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from diffusers import StableDiffusionPipeline
import torch
import gc

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Local Imports
from .state import STORE  # Import the store we just made

#  CONFIG 
DB_PATH = './chroma_db'
IMG_DIR = "generated_images"
os.makedirs(IMG_DIR, exist_ok=True)

#  Helper functions
def init_llm():
    return ChatOpenAI(
        base_url="http://localhost:1234/v1", 
        api_key="lm-studio",
        model="qwen/qwen3-vl-8b", 
        temperature=0.1
    )

def load_vectorstore():
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=emb_model, collection_name="video_rag")

def generate_local_image(prompt: str, output_path: str):
    # (Your exact diffusion code from the notebook)
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True,
            cache_dir="./huggingface_cache", low_cpu_mem_usage=True
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        image = pipe(prompt).images[0]
        image.save(output_path)
        
        # Cleanup
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"Gen Error: {e}")
        return False

# TOOLS 

# RAG Search Tool
@tool
def lookup_video_context(query: str):
    """Find facts, summaries, or quotes from the video."""
    # 1. Get the current Video ID from the Global Store
    df = STORE.get_df()
    if df is None or df.empty:
        return "Error: No video loaded."
    
    # grab the video_id from the first row of data
    current_video_id = df.iloc[0]['video_id']

    # 2. Load VectorStore with a FILTER
    vectorstore = load_vectorstore()
    
    # This filter ensures we ONLY see chunks from the current video
    retriever = vectorstore.as_retriever(
        search_kwargs={
            'k': 5, 
            'filter': {'video_id': current_video_id}
        }
    )
    
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


# Stats Analyzer Tool
@tool
def analyze_video_stats(query: str):
    """Calculate statistics or counts from the transcript."""
    df = STORE.get_df()
    if df is None or df.empty:
        return "Error: No video loaded."
        
    def count_word(target_word):
        if not target_word: return 0
        return df['text'].str.count(f'(?i){re.escape(target_word)}').sum()

    llm = init_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Python Data Analyst.
        User Question: "{query}"
        Helper available: `count_word(word_string)`
        Rules:
        1. Write python code.
        2. Do NOT import pandas.
        3. Print the result.
        """
    )
    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({"query": query}).replace("```python", "").replace("```", "").strip()
    
    # Execute
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        exec(code, {}, {"df": df, "count_word": count_word, "pd": pd, "re": re})
        result = new_stdout.getvalue()
    except Exception as e:
        result = f"Error: {e}"
    finally:
        sys.stdout = old_stdout
    return result

# Data Visual Generator Tool
@tool
def generate_trend_plot(query: str):
    """
    Use this tool ONLY when the user asks for VISUALS, PLOTS, GRAPHS, or CHARTS.
    """
    # 1. Access Data
    df = STORE.get_df()
    if df is None or df.empty:
        return "Error: No data loaded. Please ingest a video first."

    # 2. Define Helper for the LLM
    def save_plot_to_disk():
        # Ensure the directory exists
        os.makedirs(IMG_DIR, exist_ok=True)
        filename = f"{IMG_DIR}/plot_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(filename)
        plt.close()
        return filename

    # 3. Initialize LLM
    llm = init_llm()

    # 4. FULL PROMPT (Copied from your Notebook)
    prompt = ChatPromptTemplate.from_template(
    """
    You are a Python Data Visualization Agent.
    Think before calling any tool. Find out the **CORRECT PLOT** type from the user query.
    
    CONTEXT:
    - A pandas DataFrame named `df` IS ALREADY DEFINED in the environment.
    - Columns: ['seconds' (int), 'text' (str), 'sentiment' (float)]
    - You must use `df` directly.
    
    User Request: "{query}"
    
    GOAL:
    Write a Python script to generate the plot using `matplotlib.pyplot` as `plt`.
    
    ### REQUIRED CODE PATTERNS (COPY THESE EXACTLY):
    
    # Pattern 1: Simple Line Plot (Word Count over Time)
    df['matches'] = df['text'].str.lower().str.count('word_name')
    plt.figure(figsize=(10, 5))
    plt.plot(df['seconds'], df['matches'], label='word_name')
    plt.legend()
    plt.title('Mentions of word_name over time')
    print(save_plot_to_disk())

    # Pattern 2: Grouped Bar Chart (Compare Multiple Words)
    # Use offsets to place bars side-by-side
    width = 10
    df['c1'] = df['text'].str.lower().str.count('word1')
    df['c2'] = df['text'].str.lower().str.count('word2')
    
    plt.figure(figsize=(10, 5))
    plt.bar(df['seconds'] - width/2, df['c1'], width=width, label='word1', alpha=0.7)
    plt.bar(df['seconds'] + width/2, df['c2'], width=width, label='word2', alpha=0.7)
    plt.legend()
    plt.title('Comparison: Word1 vs Word2')
    print(save_plot_to_disk())

    # Pattern 3: Pie Chart (Total Counts)
    v1 = df['text'].str.lower().str.count('word1').sum()
    v2 = df['text'].str.lower().str.count('word2').sum()
    
    plt.figure(figsize=(6, 6))
    plt.pie([v1, v2], labels=['Word1', 'Word2'], autopct='%1.1f%%')
    plt.title('Total Word Frequency')
    print(save_plot_to_disk())

    ### STRICT RULES:
    1. DO NOT import pandas or create a DataFrame. `df` exists.
    2. DO NOT output markdown. Just the code.
    3. ALWAYS end with `print(save_plot_to_disk())`.
    """
    )

    # 5. Generate Code
    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({"query": query})
    
    # 6. Sanitize Code
    code = code.replace("```python", "").replace("```", "").strip()
    
    # Filter out dangerous lines (re-defining data)
    filtered_lines = []
    for line in code.split('\n'):
        if "data =" in line or "pd.DataFrame" in line or "df =" in line or "import pandas" in line:
            continue
        filtered_lines.append(line)
    code = "\n".join(filtered_lines)

    # 7. Execute
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        exec(
            code,
            {},
            {
                "df": df,
                "plt": plt,
                "save_plot_to_disk": save_plot_to_disk
            }
        )
        result = sys.stdout.getvalue().strip()
    except Exception as e:
        result = f"Error executing plot: {e}\nGenerated Code:\n{code}"
    finally:
        sys.stdout = old_stdout

    return result if result else "Error: Plot code ran but produced no output."

# Image suggetion tool
@tool
def get_image_suggestions(query: str):
    """Get 3 image prompts based on video context."""
    
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    docs = retriever.invoke("visual style and key scenes")
    context = docs[0].page_content if docs else "General video content"
    
    llm = init_llm()
    prompt = ChatPromptTemplate.from_template("Based on: {context}, give 3 stable diffusion prompts.")
    return (prompt | llm | StrOutputParser()).invoke({"context": context})

# Image generation Tool
@tool
def generate_image_from_prompt(prompt: str):
    """Generate an image from a text prompt."""
    filename = f"{IMG_DIR}/img_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    success = generate_local_image(prompt, filename)
    return filename if success else "Error generating image"

# Wikipedia MCP Tool
@tool
def consult_mcp_knowledge(query: str):
    """Consult external knowledge via MCP."""
    # 1. Allow nested event loops (Crucial for FastAPI)
    import nest_asyncio
    nest_asyncio.apply()
    
    # 2. Define the async logic inside a wrapper
    async def run_mcp_logic():
        server_params = StdioServerParameters(command="python", args=["wiki_server.py"])
        try:
            with open(os.devnull, 'wb') as devnull:
                async with stdio_client(server_params, errlog=devnull) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool("search_wikipedia", arguments={"query": query})
                        if result and result.content:
                            return result.content[0].text
                        return "No data found."
        except Exception as e:
            return f"MCP Error: {str(e)}"

    # 3. Force it to run synchronously
    # This blocks the graph until the result is ready, preventing the "Sync not supported" error.
    return asyncio.run(run_mcp_logic())

# Export list for the graph
ALL_TOOLS = [
    lookup_video_context, 
    analyze_video_stats, 
    generate_trend_plot, 
    get_image_suggestions, 
    generate_image_from_prompt,
    consult_mcp_knowledge
]