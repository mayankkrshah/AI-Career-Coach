# 💼 AI Career Coach

## 🚀 Empowering Your Career Journey with AI

An AI-powered career coaching platform with intelligent query routing, job database integration, and web search capabilities. Currently in active development with core chat functionality and data integration implemented.

## ✨ Current Features

- **💬 AI-Powered Chat Interface**: Interactive career coaching conversations with intelligent query routing
- **📊 Job Database Integration**: Access to 15,000+ job positions with salary and skills data
- **🔍 Smart Query Routing**: Automatically determines whether to use job database, web search, or general AI advice
- **🌐 Web Search Integration**: Real-time information via Tavily API for current market trends
- **📚 Research Paper Access**: ArXiv integration for AI/ML research and academic papers
- **⚙️ Multiple AI Models**: Support for OpenAI GPT-4, with configurable API keys
- **💾 Session Management**: Save and manage multiple chat sessions
- **📱 Responsive UI**: Modern Next.js frontend with Material-UI components

## 🏗️ Architecture

- **Backend**: FastAPI with LangChain for AI agent orchestration
- **Frontend**: Next.js with TypeScript for modern, responsive UI
- **AI Framework**: LangGraph for multi-agent workflows
- **Vector Database**: Qdrant for intelligent document retrieval
- **External APIs**: OpenAI, Cohere, Tavily for enhanced AI capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- OpenAI API key
- Cohere API key (optional)
- Tavily API key (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mayankkrshah/AI-Career-Coach.git
   cd AI-Career-Coach
   ```

2. **Set up the backend**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd frontend
   npm install
   ```

4. **Configure environment variables**
   ```bash
   # Copy and configure your API keys
   cp .env.example .env
   ```

### Running the Application

1. **Start the backend**
   ```bash
   cd api
   uvicorn app:app --reload --port 8000
   ```

2. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **LangChain**: Framework for developing LLM applications
- **LangGraph**: Multi-agent workflow orchestration
- **Qdrant**: Vector database for semantic search
- **Pydantic**: Data validation and settings management

### Frontend
- **Next.js**: React framework with TypeScript
- **Tailwind CSS**: Utility-first CSS framework
- **React Query**: Data fetching and caching
- **Lucide React**: Beautiful icons

### AI & Data
- **OpenAI GPT**: Primary language model
- **Cohere**: Alternative language model and embeddings
- **Tavily**: Web search and research capabilities
- **Pandas**: Data manipulation and analysis

## 📁 Project Structure

```
AI-Career-Coach/
├── api/                    # Backend FastAPI application
│   ├── agents/            # AI agent implementations
│   ├── career_coach_app.py # Main application logic
│   ├── career_coach_rag.py # RAG implementation
│   └── requirements.txt   # Python dependencies
├── frontend/              # Next.js frontend application
│   ├── src/app/          # App router pages and components
│   ├── src/components/   # Reusable UI components
│   └── package.json      # Node.js dependencies
├── data/                 # Training data and documents
├── notebook/             # Jupyter notebooks for analysis
└── deploy.sh            # Deployment scripts
```

## 🗺️ Roadmap

### 🚧 Planned Features
- **📝 Resume Optimization**: AI-powered resume analysis and improvement suggestions
- **💬 Interactive Interview Coaching**: Practice interviews with AI feedback and scoring
- **🎯 Personalized Career Guidance**: Tailored career path recommendations
- **🤝 Networking Assistant**: LinkedIn integration and networking strategies
- **📊 Skills Assessment**: Comprehensive skill gap analysis and learning recommendations
- **💰 Salary Negotiation**: Market-based salary insights and negotiation strategies
- **📈 Career Progress Tracking**: Goal setting and progress monitoring

### 🔧 Technical Improvements
- Multi-agent system with specialized career coaching agents
- Enhanced RAG system with vector database (Qdrant)
- Real-time collaboration features
- Mobile app development
- Advanced analytics and reporting


## 🙏 Acknowledgments

- Built with inspiration from AI Makerspace community
- Powered by OpenAI, LangChain, and modern web technologies
- Special thanks to the open-source community