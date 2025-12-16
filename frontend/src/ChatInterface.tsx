import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { Send, Bot, User, Image as ImageIcon, Trash2 } from "lucide-react";

// 1. Define the shape of a Message
interface Message {
  role: "user" | "ai";
  content: string;
  image?: string | null;
}

// 2. Define the Backend Response shape
interface ChatResponse {
  response: string;
  image_url: string | null;
  transcript_loaded: boolean;
}

export default function ChatInterface() {
  // --- STATE MANAGEMENT ---
  // We tell TypeScript that 'messages' is an array of the Message interface defined above
  const [messages, setMessages] = useState<Message[]>([
    { role: "ai", content: "Hi! Paste a YouTube URL to get started." }
  ]);
  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isVideoLoaded, setIsVideoLoaded] = useState<boolean>(false);
  
  // Auto-scroll to bottom ref
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // --- AUTO SCROLL ---
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  // --- HANDLERS ---
  
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input;
    setInput(""); // Clear input
    setIsLoading(true);

    // Add User Message
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      // POST request
      const response = await axios.post<ChatResponse>("http://localhost:8000/chat", {
        message: userMessage,
      });

      const data = response.data;

      // Add AI Response
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: data.response,
          image: data.image_url,
        },
      ]);

      if (data.transcript_loaded) setIsVideoLoaded(true);

    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "⚠️ Error connecting to server. Is the backend running?" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = async () => {
    try {
        await axios.post("http://localhost:8000/clear");
        setMessages([{ role: "ai", content: "History cleared. Ready for a new video!" }]);
        setIsVideoLoaded(false);
    } catch (e) {
        console.error("Failed to clear", e);
    }
  };

  // We explicitly type the event 'e' here
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // --- RENDER ---
  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto bg-zinc-950 shadow-2xl border-x border-zinc-800">
      
      {/* HEADER */}
      <header className="bg-gradient-to-r from-[#831843] to-[#0f172a] text-white p-4 flex justify-between items-center shadow-lg border-b border-white/10">
        <div className="flex items-center gap-2">
            <Bot className="w-8 h-8" />
            <h1 className="text-xl font-bold">UTube Assistant</h1>
        </div>
        
        <div className="flex items-center gap-4">
            {isVideoLoaded && (
                <span className="bg-green-500 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider">
                    Video Loaded
                </span>
            )}
            <button 
                onClick={clearHistory}
                className="p-2 hover:bg-blue-700 rounded-full transition"
                title="Clear History"
            >
                <Trash2 className="w-5 h-5" />
            </button>
        </div>
      </header>

      {/* MESSAGES AREA */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-gray-50">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
          >
            {/* Avatar */}
            <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 border ${
              msg.role === "ai" 
              ? "bg-white text-slate-800 border-slate-200 shadow-sm"  // AI Avatar
              : "bg-slate-900 text-white border-slate-900 shadow-sm"   // User Avatar
              }`}>
                {msg.role === "ai" ? <Bot size={24} /> : <User size={24} />}
            </div>

            {/* Bubble */}
            <div className={`max-w-[80%] rounded-2xl p-5 shadow-sm leading-relaxed ${
            msg.role === "ai" 
            ? "bg-white text-slate-800 border border-gray-100 shadow-md" 
            : "bg-slate-900 text-gray-50" // Deep Slate User Bubble
            }`}>
              
              {/* Text Content */}
              <div className="prose prose-sm max-w-none dark:prose-invert">
                 <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>

              {/* Image Content */}
              {msg.image && (
                <div className="mt-4">
                    <img 
                        src={msg.image} 
                        alt="Generated Chart" 
                        className="rounded-lg border border-gray-200 shadow-sm max-h-80 object-contain bg-white"
                    />
                    <div className="flex items-center gap-1 text-xs text-gray-400 mt-2">
                        <ImageIcon size={12} /> Generated by UTube Assistant
                    </div>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {/* Loading Indicator */}
        {isLoading && (
            <div className="flex gap-4">
                <div className="w-10 h-10 bg-white border border-slate-200 rounded-full flex items-center justify-center shadow-sm">
                  <Bot size={24} className="text-slate-800" />
                </div>
                <div className="bg-white border p-4 rounded-2xl shadow-sm">
                    <div className="flex gap-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                    </div>
                </div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* INPUT AREA */}
      <div className="p-4 bg-white border-t border-gray-100">
        <div className="flex gap-2 relative">
          <input
            type="text"
            className="flex-1 p-4 pr-12 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50 focus:bg-white transition-all shadow-sm"
            placeholder="Paste a YouTube link or ask a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="absolute right-2 top-2 bottom-2 bg-slate-900 text-white p-3 rounded-xl hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg"
          >
            <Send size={20} />
          </button>
        </div>
        <p className="text-center text-xs text-gray-400 mt-2">
            AI can make mistakes. Please verify important information.
        </p>
      </div>
    </div>
  );
}