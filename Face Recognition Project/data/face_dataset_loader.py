from sklearn.datasets import fetch_olivetti_faces

def load_olivetti_faces():
    """Load the Olivetti faces dataset."""
    faces = fetch_olivetti_faces()
    return faces.images

