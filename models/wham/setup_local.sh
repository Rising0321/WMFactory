# Tested using Python 3.9

echo "Making and activating a new virtual environment..."
python3.9 -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Instaling the required packages..."
pip install -r requirements.txt

echo "Instaling the exiftool package for adding file metadata on Linux..."
sudo apt install -y exiftool

echo "Installing ffmpeg..."
sudo apt install ffmpeg

echo "All packages installed successfully!"
