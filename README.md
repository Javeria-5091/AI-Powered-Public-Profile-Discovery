# AI-Powered Public Profile Discovery Module

An AI-powered profile discovery system built with FastAPI that allows users to search by industry, field, or keyword (e.g., AI Development, Machine Learning, Data Science). The system generates AI-enhanced search queries, discovers public GitHub profiles, ranks them by relevance, and stores search history and results in a SQLite database.

## Features

- AI-based query generation
- Public profile discovery via GitHub API
- Relevance ranking system
- Search history tracking
- SQLite database integration
- REST APIs with FastAPI

## APIs

- POST /api/profile-search
- GET /api/profile-search/history
- GET /api/profile-search/{id}

## Technologies Used

- Python
- FastAPI
- SQLite
- SQLAlchemy
- GitHub API
- OpenAI / GitHub Models
