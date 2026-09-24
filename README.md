## 🚀 Key Features

*   **Conversational AI Agent**: An intelligent chatbot that collects casting requirements (location, budget, shoot dates, etc.) and performs real-time talent searches.
*   **Stateful Workflows**: Built with LangGraph to handle complex, multi-step casting logic and mandatory field collection.
*   **Job Management**: Create casting calls, manage drafts, and track applicant counts.
*   **Self-Tape & Polas System**: 
    *   Request self-tapes (videos) or polaroids (images) from talent.
    *   Dedicated endpoints for talent to upload media files or provide external URLs.
    *   Robust status tracking (Requested, Responded, Accepted, Rejected).
*   **Shortlisting & Booking**: Streamlined workflow to shortlist talent and manage final bookings.
*   **Media Serving**: Integrated static file serving for high-performance access to talent images and videos.
*   **Secure Authentication**: JWT-based authentication integrated with a main account system.

## 🛠 Tech Stack

*   **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
*   **AI Engine**: [OpenAI GPT-5.1](https://openai.com/), [LangChain](https://www.langchain.com/), [LangGraph](https://python.langchain.com/docs/langgraph)
*   **Database**: PostgreSQL (via [NeonDB](https://neon.tech/))
*   **ORM**: [SQLAlchemy](https://www.sqlalchemy.org/)
*   **Authentication**: JWT (Jose)
*   **Environment**: Python 3.9+

## 📋 Prerequisites

*   Python 3.9 or higher
*   A NeonDB (or PostgreSQL) instance
*   OpenAI API Key

## ⚙️ Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd CastLink-AI
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the `app/` directory:
    ```env
    DATABASE_URL=postgresql://user:password@host/dbname
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_CHAT_MODEL=gpt-5.1
    JWT_SECRET_KEY=your_secret_key
    JWT_ALGORITHM=HS256
    ```

## 🏃 Running the Application

Start the FastAPI server using Uvicorn:

```bash
python app/main.py
```
The server will start at `http://0.0.0.0:8008`.

## 📁 Project Structure

```text
app/
├── main.py          # FastAPI entry point and route definitions
├── database.py      # SQLAlchemy models and database configuration
├── schemas.py       # Pydantic models for request/response validation
├── services.py      # LangGraph logic, AI tools, and utility functions
├── auth.py          # JWT authentication dependencies
├── media/           # Local storage for uploaded selftapes and polas
└── .env             # Environment configuration
```
