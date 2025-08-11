python -m venv venv
source venv/Scripts/activate

python -r requirements.txt

#Docker qdrant:
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}\qdrant_data:/qdrant/storage qdrant/qdrant:latest

streamlit run app.py