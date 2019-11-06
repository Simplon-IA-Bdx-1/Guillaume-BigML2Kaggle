# Les étapes à respecter

## 1ère partie : Les Features

1.1. Ouvrir les dataframes

```
    fulltrain=read_csv('./cs-training.csv',index_col=0)
    test=read_csv('./cs-test.csv',index_col=0)

#index_col=0 la colonne qu'on doit utiliser comme index dans le fichier csv
```

1.2. Faire le split sur le trainfull (à faire qu’une seule fois)

Il faut choisir un seed pour faire un split identique à chaque fois (sinon c'est random) :
```
    train_dataset = api.create_dataset(
        origin_dataset, {"name": "Dataset Name | Training",
                        "sample_rate": 0.8, "seed": "my seed"})
    test_dataset = api.create_dataset(
        origin_dataset, {"name": "Dataset Name | Test",
                        "sample_rate": 0.8, "seed": "my seed",
                        "out_of_bag": True})
```

1.3. Modifier les features à la fois sur le train, valid, test.

Mettre les 3 dans un tableau et faire une boucle

    ex :
```
    data_sets = [fulltrain, train80, valid20, test]
    for df in data_sets :
        modifier df
```

1.4. Sauvegarder en fichier .csv les df(train80, valid20, test sets) :
```
    train80.to_csv(train80.csv,index_label='Id')
```

## 2ème Partie : Créer le modèle

2.1. Une fois les features modifiés, il faut envoyer les sources (csv) sur BIgML.

Ceci est à faire pour les fichiers train, validation et test :

```
train_src = api.create_source(train_filename) #créer les sources

api.ok(train_src) # Met le programme en pause le temps que BigML finisse de télécharger

train_ds = api.create_dataset(train_src) #créer les datasets
api.ok(train_ds)
```

2.2.  Maintenant qu'on a créé nos datasets, il faut créer nos modèles (à faire que sur le train car c'est le train qui nous aide à faire le modèle) :

```
model_args= {"objective_field": "SeriousDlqin2yrs"} #c'est là où on indique l'output field (=l'objet de la prédiction)

model = api.create_ensemble(train_ds, model_args)
api.ok(model)
```


## 3ème Partie : L'évaluation

L'évaluation se fait sur la validation 'valid20_ds' et sur le model 'model'.

Deux méthodes pour l'évaluation :

* Evaluation sur BigMl :
```
evaluation = api.create_evaluation(model, valid_ds)
pprint(evaluation) #pprint = pretty print
```
    Dans cette étape, on a la matrice de confusion, l'AUC.


* Evaluation en Python avec des fonctions à appeler :

    * 1ère étape : Faire un Batch Prediction
```
batch_prediction = api.create_batch_prediction(model, valid_ds,
    {
    "name": "prediction.csv",   #nom du fichier dans lequel enregistrer la prédiction
    "all_fields": True,     #garder tous les champs (si on veut analyser les erreurs/features)
    "header": True,     #avoir la 1ère ligne des noms de colonnes
    "confidence": True,     #nous donne l'indice de confiance entre 0,5 et 1
    "probabilities": True       #colonnes de '0 proba' et '1 proba' mais ce qui nous intéresse c'est la colonne '1 proba'
    }                      )
api.ok(batch_prediction)
```
    Ouvrir le csv et le transformer pour avoir un dataframe avec la fonction read_csv (prediction_df).

* 
    * 2ème étape : Calcul de la matrice de confusion avec pandas_ml
```
from pandas_ml import ConfusionMatrix

confusion_matrix = ConfusionMatrix(prediction['SeriousDlqin2yrs'],prediction['SeriousDlqin2yrs.1'])

d = {'P\u0302': [confusion_matrix.TP, confusion_matrix.FP, confusion_matrix.PPV],
     'N\u0302': [confusion_matrix.FN, confusion_matrix.TN, confusion_matrix.NPV],
     'Recall': [confusion_matrix.TPR, confusion_matrix.TNR, confusion_matrix.ACC]}
confusion_matrix_df = DataFrame(data=d,index=['P','N', 'Precision'])
confusion_matrix_df
```
* 
    * 3ème étape : L'analyse de gain/coût

Attention pseudo-code:

```
Trier le dataframe selon la colonne '1 proba'
Calculer les seuils intermédiaires : 
    Copier la colonne '1 proba' en '1 proba b'
    Décaler la colonne '1 proba b' de 1 vers le bas (fonction shift(fill_value=0))
    Créer colonne seuil qui est la moyenne de '1 proba' et '1 proba b'
Pour chaque seuil :
    Calculer le nombre d'élément dans chaque classe : TP, TN, FP, FN
    Calculer le coût en multipliant le nombre d'élements par leur coût correspondant
    Enregistrer le résultat dans une colonne coût #ici on a calculé le gain pour chaque seuil intermédiaire

On cherche l'index qui correspond au meilleur gain (idxmax) :

max_index = cost_df['cost'].idxmax() #index qui correspond au gain maximum
max_gain = cost_df.loc[max_index]['cost'] # gain maximum
max_threshold = cost_df.loc[max_index]['threshold'] #seuil optimal
```
* 
    * 4ème étape :
        * Calcul à la main de l'AUC

Attention pseudo-code:
```
score=0
Pour chaque positif P:
    Pour chaque négatif N:
        Si '1 proba' de P > '1 proba' de N:
            score+=1
AUC = score/(nombre de N * nombre de P)
```
```
nombre de TP = 0
score = 0
Trier selon '1 proba' décroissant

Pour chaque ligne:
    Si la ligne est positive:
        nombre de TP +=1
    Sinon:
        score += nombre de TP
AUC = score/(nombre de N * nombre de P)
```
* 
    * 
        * Calcul de l'AUC avec sklearn
```
from sklearn.metrics import roc_curve, auc, roc_auc_score
AUC = score/(nombre de N * nombre de P)
```


## 4ème Partie : Analyse d'erreurs

4.1. Trier les prédictions par erreur
```
prediction_df['absolute_error']=(prediction_df['1 probability']-prediction_df['SeriousDlqin2yrs']).abs()
# différence entre la prédiction et la vraie valeur (en valeur absolue)

prediction_errors = prediction_df.sort_values(by='absolute_error', ascending=False).head(100)
#on trie par le niveau d'erreur et on sélectionne les 100 plus grosses erreurs
```
4.2. On sépare ensuite les FP et FN.

4.3. On analyse les erreurs en essayant de comprendre pourquoi on s'est trompé.


## 5ème Partie : Identifier les axes d'amélioration et reprendre à l'étape 1

## 6ème Partie : Soumettre à Kaggle

```
kaggle_prediction_df=DataFrame(index=prediction_df.index)
#On créé un dataset en copiant la colonne index car sur Kaggle il faut que les numéros correspondent

kaggle_prediction_df['Probability']=prediction_df['1 probability']
#On copie la colonne 1 probability qu'on renomme Probability pour Kaggle. 

kaggle_prediction_file="kaggleprediction.csv"
#On définit un nom de fichier csv qu'on envoie à Kaggle

kaggle_prediction_df.to_csv(kaggle_prediction_file,index_label='Id')
#On enregistre tout le dataframe dans le fichier qu'on a définit. On utilise la colonne index comme colonne Id.

kaggle.api.competition_submit(kaggle_prediction_file, "commentaire sur ma prediction", "GiveMeSomeCredit")
#On envoie la prédiction sur Kaggle. (1er arg = nom_du_fichier, 2e arg= commentaire ou nom de la soumission, 3e arg = nom du challenge)
```