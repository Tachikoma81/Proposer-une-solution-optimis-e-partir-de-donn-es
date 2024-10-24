import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gradio as gr

# Fonction pour récupérer les documents et transformer en matrice TF-IDF
def utile():
    # Chemin vers le répertoire contenant les fichiers texte
    directory = 'fini'  # Assure-toi que ce chemin existe

    # Liste pour stocker le contenu des fichiers
    documents = []
    document_names = []

    # Parcourir chaque fichier dans le répertoire et lire son contenu
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                document_names.append(filename)  # Ajouter les noms de fichier pour les lignes

    # Initialisation du vectoriseur TF-IDF
    vectorizer = TfidfVectorizer()

    # Ajuster le modèle TF-IDF et transformer les documents en une matrice TF-IDF
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Convertir la matrice TF-IDF en DataFrame pandas
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=document_names, columns=vectorizer.get_feature_names_out())
    return df_tfidf,document_names

# Fonction pour trouver les meilleurs documents pour un terme donné
def top_n_documents_for_term(term, n, df):
    if term in df.columns:
        top_docs = df[term].sort_values(ascending=False).head(n)
        return top_docs.to_string()  # Convertir en chaîne de caractères pour affichage
    else:
        return f"Le mot '{term}' n'est pas présent dans les documents."

# Fonction pour la partie "listing"
def listing_function(name):
    return str(name)  # Retourner les résultats sous forme de chaîne de caractères

# Fonction pour la partie "recherche"
def recherche_function(term, n, df):
    return top_n_documents_for_term(term, n, df)

# Fonction pour la partie "recommendation"
def recommendation_function():
    return "Recommendation functionality goes here."

# Charger les données une seule fois
df,name = utile()

# Interface principale Gradio avec des onglets
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Listing"):
            gr.Markdown("## Page Listing")
            bouton_listing = gr.Button("Lister", variant="primary")
            output_listing = gr.Textbox(label="Résultat")
            bouton_listing.click(fn=lambda: listing_function(name), outputs=output_listing)

        with gr.TabItem("Recherche"):
            gr.Markdown("## Page Recherche")
            mots = gr.Textbox(label="Entrer vos mots")
            serie = gr.Number(label="Nombre de résultats (entre 10 et 30)", value=10, precision=0)
            bouton_recherche = gr.Button("Valider recherche")
            output_recherche = gr.Textbox(label="Résultat de la recherche")
            bouton_recherche.click(fn=lambda term, n: recherche_function(term, n, df), inputs=[mots, serie], outputs=output_recherche)

        with gr.TabItem("Recommendation"):
            gr.Markdown("## Page Recommendation")
            bouton_recommendation = gr.Button("Recommander", variant="primary")
            output_recommendation = gr.Textbox(label="Résultat")
            bouton_recommendation.click(fn=recommendation_function, outputs=output_recommendation)

# Lancer l'interface
demo.launch()
