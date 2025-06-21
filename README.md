# Image Quality Assessment Application

This repository contains a deep learning-based Image Quality Assessment (IQA) system that predicts Mean Opinion Scores (MOS) for images. The application includes both a FastAPI backend and a Streamlit frontend interface.

## 🌟 Features

- Upload and analyze image quality
- Real-time quality metrics calculation
- Image enhancement capabilities
- Visual quality score representation
- Downloadable enhanced images
- Docker support for easy deployment

## 🛠️ Prerequisites

- Python 3.10+
- Docker (optional)
- Git

## 🚀 Quick Start

### Method 1: Direct Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Quality-Assessment.git
cd Image-Quality-Assessment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r req_app_env1.txt
```

4. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

### Method 2: Using Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Quality-Assessment.git
cd Image-Quality-Assessment
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

## 📊 Accessing the Application

- Streamlit Frontend: `http://localhost:8501`
- FastAPI Backend: `http://localhost:8000`

## 📁 Project Structure

```
├── cnn_model.py           # CNN model architecture and training
├── feature_extraction.py  # Image feature extraction utilities
├── main.py               # FastAPI backend
├── streamlit_app.py      # Streamlit frontend
├── docker-compose.yml    # Docker compose configuration
├── Dockerfile.api        # API Dockerfile
├── Dockerfile.streamlit  # Streamlit Dockerfile
└── requirements.txt      # Project dependencies
```

## 🔧 Configuration

The application can be configured through environment variables:
- `PORT`: API port (default: 8000)
- `HOST`: API host (default: 0.0.0.0)
- `MODEL_PATH`: Path to the trained model file

## 📝 API Documentation

When running the FastAPI backend, visit:
- API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

