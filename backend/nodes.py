import pandas as pd
import yt_dlp
import datetime
from textblob import TextBlob
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat, _parse_video_id
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, SystemMessage

# Import from our other files
from .state import AppState, STORE
from .tools import load_vectorstore, init_llm, ALL_TOOLS

# HELPERS FOR YOUTUBE 

class YtDlpYoutubeLoader(YoutubeLoader):
    """Custom Loader to handle YouTube headers and metadata robustly."""
    def _get_video_info(self):
        ydl_opts = {"quiet": True}
        # Configure yt_dlp to avoid some bot detection
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={self.video_id}", 
                    download=False
                )
                return {
                    "title": info.get("title", "Unknown"),
                    "description": info.get("description", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "thumbnail_url": info.get("thumbnail", "Unknown"),
                    "publish_date": info.get("upload_date", "Unknown"),
                    "length": info.get("duration", 0),
                    "author": info.get("uploader", "Unknown"),
                }
            except Exception as e:
                print(f"Error fetching metadata: {e}")
                return {}

def format_timestamp(seconds) -> str:
    """Converts 125.0 to '02:05'."""
    try:
        return str(datetime.timedelta(seconds=int(seconds)))
    except:
        return str(seconds)

#  NODES 

def ingest_video_node(state: AppState):
    """Node 1: Downloads and Indexes the Video."""
    url = state.get('youtube_url')
    if not url: 
        return {"messages": [AIMessage(content="Error: No YouTube URL found.")]}

    try:
        video_id = _parse_video_id(url)
    except Exception:
        video_id = url 

    print(f"--- Ingesting Video: {video_id} ---")
    # Remove the old video
    STORE.set_df(None)
    
    vectorstore = load_vectorstore()
    # Check if video exists in DB
    existing_data = vectorstore.get(where={"video_id": video_id})
    structured_data = []

    # CASE A: Load from DB
    if len(existing_data['ids']) > 0:
        print("Video found in DB. Loading from cache...")
        for text, metadata in zip(existing_data['documents'], existing_data['metadatas']):
            score = TextBlob(text).sentiment.polarity
            structured_data.append({
                "timestamp": metadata.get('timestamp_str', '00:00:00'),
                "seconds": metadata.get('start_seconds', 0),
                "text": text,
                "sentiment": score,
                "video_id": video_id
            })
            
    # CASE B: Download New
    else:
        print(f"Downloading new video: {url}...")
        try:
            loader = YtDlpYoutubeLoader.from_youtube_url(
                url, add_video_info=True, transcript_format=TranscriptFormat.CHUNKS, 
                chunk_size_seconds=60,
                language=["en", "en-US", "en-GB", "en-CA", "en-AU"]
            )
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            for doc in docs:
                ts = format_timestamp(doc.metadata.get('start_timestamp', 0))
                doc.metadata['video_id'] = video_id
                doc.metadata['timestamp_str'] = ts
                
                video_title = doc.metadata.get('title', 'Unknown Video')
                start_seconds = doc.metadata.get('start_timestamp', 0)
                timestamp = format_timestamp(start_seconds)
                
                # Sentiment & Data Structuring
                score = TextBlob(doc.page_content).sentiment.polarity
                structured_data.append({
                    "timestamp": ts,
                    "seconds": doc.metadata.get('start_seconds', 0),
                    "text": doc.page_content,
                    "sentiment": score,
                    "video_id": video_id
                })
                
                # Context Tagging for RAG
                doc.page_content = f"[Video: {video_title}] [Time: {timestamp}]\n\n{doc.page_content}"
            
            # Save to Chroma
            splits = text_splitter.split_documents(docs)
            vectorstore.add_documents(documents=splits)
            print(f"Saved {len(splits)} chunks to VectorStore.")

        except Exception as e:
            return {"messages": [AIMessage(content=f"Failed to load video: {str(e)}")]}
    
    #Update the Global Store
    df = pd.DataFrame(structured_data)
    STORE.set_df(df)
    print(f"Global DataFrame Updated: {len(df)} rows.")
    
    success_message = (
        f"Video processed successfully! I've loaded the transcript and data for '{video_id}'. "
        "What would you like to know? (e.g., 'Summarize it', 'Anlyze the content?', 'Show me a plot')"
    )

    return {
        "is_processed": True,
        "transcript_structure": structured_data.append,
        "messages": [AIMessage(content=success_message)]
    }

def agent_node(state: AppState):
    """Node 2: The AI Manager."""
    print("--- Agent Thinking ---")
    messages = state['messages']
    
    # 1. Check Context via STORE
    video_title = "No Video Loaded"
    is_loaded = False
    
    df = STORE.get_df()
    if df is not None and not df.empty:
        is_loaded = True
        try:
            first_text = df.iloc[0]['text']
            if "[Video:" in first_text:
                video_title = first_text.split("]")[0].replace("[Video:", "").strip()
        except:
            video_title = "Unknown"

    # 2. DEFINE THE PROMPTS
    # We use a 'Base' prompt for personality/safety, and switch instructions based on state.
    
    base_identity = (
        "You are 'TubeBot', a friendly, engaging, and helpful AI Video Assistant.\n"
        "Your tone should be conversational and encouraging (e.g., 'Sure thing!', 'I found this...').\n\n"
        
        "### SAFETY & CONDUCT GUIDELINES:\n"
        "1. **Harmful Content:** Strictly REFUSE to generate, summarize, or analyze content related to hate speech, self-harm, explicit violence, or sexual material.\n"
        "2. **Politeness:** If a user is rude, remain polite and professional.\n"
        "3. **Boundaries:** If asked about topics outside the video or general knowledge (unless using the MCP tool), politely clarify you are focused on the video content.\n"
    )

    if not is_loaded:
        # SCENARIO A: NO VIDEO LOADED 
        # We force the agent to ONLY ask for a URL.
        sys_msg = (
            f"{base_identity}\n"
            "### CURRENT STATUS: NO VIDEO LOADED\n\n"
            
            "### YOUR PRIMARY GOAL:\n"
            "You are currently waiting for the user to provide a YouTube URL.\n"
            "1. If the user says 'Hi', 'How to start', or asks what you can do, explain that you need a YouTube link to begin.\n"
            "2. **CRITICAL:** Do NOT ask for topics, descriptions, or titles. You cannot search YouTube. You can ONLY process a direct URL.\n"
            "3. **Example Response:** 'Hi there! To get started, please paste a YouTube URL here, and I'll analyze it for you!'\n"
        )
    else:
        # SCENARIO B: VIDEO IS LOADED 
        # We give the agent full access to tools.
        sys_msg = (
            f"{base_identity}\n"
            f"### STATUS: VIDEO LOADED: '{video_title}'\n\n"
            
            "### RESPONSE STYLE (MUST FOLLOW):\n"
            "1. **Be Direct:** If a tool fails and you retry (or if you find the wrong info first), **DO NOT** mention the error or the process. Just state the final correct answer.\n"
            "2. **No Meta-Commentary:** Avoid phrases like 'I accessed the wrong file...', 'It seems I cannot access specific...', or 'My apologies'.\n"
            "3. **Just the Facts:** Simply say: 'LLM is about ...'\n"
            "4. **CRITICAL:** If you are not sure about the answer ask further question to user rather giving wrong output'\n"
            
            "### DECISION PROTOCOL (EXECUTE IN ORDER):\n"
            "**STEP 1: CHECK THE VIDEO (ALWAYS FIRST)**\n"
            "   - Use `lookup_video_context` to find what the video says about the topic.\n"
            "   - *Decision Point:* Did the video answer the question *fully*?\n"
            "       - **YES (Full Bio/Answer):** Answer using ONLY the video.\n"
            "       - **PARTIAL (Mentioned only):** If the video mentions a name (e.g., 'Grealish scores') but doesn't explain *who* they are, proceed to STEP 2.\n"
            "       - **NO (Not found):** Proceed to STEP 2.\n\n"

            "**STEP 2: ENRICH WITH EXTERNAL KNOWLEDGE (If needed)**\n"
            "   - If the video context was missing or insufficient (as per Step 1), use `consult_mcp_knowledge`.\n"
            "   - *Rule:* Explicitly state: 'The video mentions [X], but to give you more context, I'll look up who that is...'\n\n"
            
            "**STEP 3: VISUALS (If asked)**\n"
            "   - Use `generate_trend_plot` for data/charts.\n"
            "   - Use `get_image_suggestions` for thumbnails/art.\n\n"
            
            "### EXAMPLES:\n"
            "- User: 'Who is Jack?' -> Video says: 'Jack passes.' -> **Action:** Video has name, but no bio. Call `consult_mcp_knowledge` to enrich.\n"
            "- User: 'What did Jack do?' -> Video says: 'Jack passes.' -> **Action:** Answer from video ('He passed the ball').\n"
            
            
        )


    # 3. Bind Tools
    llm = init_llm()
    if is_loaded:
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
    else:
        llm_with_tools = llm 

    # 4. Invoke
    response = llm_with_tools.invoke([SystemMessage(content=sys_msg)] + messages)
    return {"messages": [response]}