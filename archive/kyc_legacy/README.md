# Archive KYC / CIN

Ce dossier contient les anciens composants de vérification d’identité qui ne
font pas partie de l’API de validation des preuves de paiement.

## Fonctions archivées

- lecture OCR de la CIN mauritanienne ;
- extraction du NNI et des informations d’identité ;
- comparaison de visages ;
- détection de pose et de présence réelle ;
- routes Gemini consacrées à la CIN/KYC ;
- modèles CIN historiques.

Ces fichiers sont conservés pour une éventuelle réutilisation future, mais
ils ne doivent pas être importés par l’API de validation des paiements.

## Objectif actif du dépôt

Le code actif doit se concentrer sur :

- Bankily ;
- Masrivi ;
- Sadad ;
- Click ;
- qualité de la capture ;
- détection d’une preuve de paiement ;
- OCR du montant, du statut et de la référence ;
- comparaison avec le montant attendu ;
- détection des preuves réutilisées.
