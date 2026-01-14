import urllib.request
import zipfile
import os

url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
zip_filename = "adult.zip"

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def download_dataset():
    # Télécharger le fichier
    print("Téléchargement du dataset...")
    urllib.request.urlretrieve(url, zip_filename)

    # Décompresser le fichier
    print("Décompression...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(".")

    # Supprimer l'archive zip (optionnel)
    os.remove(zip_filename)

    print("Terminé ! Le dataset est prêt.")


if __name__ == "__main__":
    download_dataset()