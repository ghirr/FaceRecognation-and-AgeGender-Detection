#Projet :
Ce projet permet de créer un système de reconnaissance faciale simple et la détection d’âge et de genre en utilisant la caméra d'un ordinateur . Il démontre le processus de capture d'images pour créer un dataset, l'entraînement d'un modèle k-NN, et la reconnaissance faciale en temps réel.

#Technologies utilisées :
##	OpenCV (Open Source Computer Vision Library) :
OpenCV est une bibliothèque open-source de vision par ordinateur et de traitement d'image. Elle fournit des outils puissants pour la manipulation d'images, la détection d'objets, le suivi, etc.
Fonctionnalité dans le projet : Utilisé pour la capture vidéo à partir de la caméra, la détection de visages à l'aide de classificateurs en cascade (haarcascades), et la manipulation d'images.

##	NumPy (Numerical Python) :
NumPy est une bibliothèque Python pour effectuer des calculs numériques. Elle fournit des tableaux multidimensionnels, des fonctions pour effectuer des opérations sur ces tableaux, et des outils pour intégrer du code C/C++ et Fortran.
Fonctionnalité dans le projet : Utilisé pour le stockage et la manipulation des données d'image, notamment dans la partie de collecte des données d'entraînement.


##	KNN (k-Nearest Neighbors) :
Le k-NN est un algorithme d'apprentissage supervisé simple. Il fonctionne en trouvant les k échantillons les plus proches dans un ensemble de données pour prédire la classe ou la valeur d'un nouvel échantillon.
Fonctionnalité dans le projet : Utilisé pour la reconnaissance faciale et la détection d’âge et de genre. L'algorithme k-NN est appliqué sur les données d'entraînement pour prédire la classe (identité) des visages détectés en temps réel.

##	Haarcascades (Haar Cascade Classifiers) :
Les classificateurs en cascade Haar sont des modèles utilisés pour la détection d'objets dans des images. Ils sont largement utilisés pour la détection de visages.
Fonctionnalité dans le projet : Utilisé dans les scripts de détection et de collecte de visages pour identifier les zones de l'image qui contiennent des visages
##	Time :
La bibliothèque time offre des fonctionnalités pour travailler avec le temps. Elle expose différentes fonctions pour mesurer le temps écoulé, manipuler des objets de temps, et introduire des délais dans l'exécution du programme.
Fonctionnalités dans le projet :Dans le projet de reconnaissance faciale, la bibliothèque time est utilisée pour mesurer le temps d'exécution de certaines parties du code. Plus précisément, vous utilisez la fonction time.time() pour mesurer la durée entre deux points dans le code. Cela peut être utile pour évaluer les performances de différentes opérations, notamment la détection de visages, la reconnaissance faciale, ou d'autres tâches chronophages..
## Real-time-Face-Recognition-Project

Steps to run 
1. First run video read.py to check that your webcam is running or not.
2. Run face_detection.py to check that whether camera is able to capture your face or not, with the help of haar cascase classifier
3. Now, run face_data.py --> this will open up the camera and extract your face from video frames multiple time. Test and store faces of multiple people.
4. At last, run face_recognition.py --> this will detect your face from the dataset made and form a bounding box with you name writtern around your face.
