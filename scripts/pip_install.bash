eval "$(pyenv init -)"
pip install --upgrade pip
# Specify path from the project root (not from this file)
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_dev.txt 
# NLP Kits
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
