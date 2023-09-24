### CSP

Le CSP, ou "Common Spatial Patterns" (Modèles spatiaux communs), est une technique d'extraction de caractéristiques largement utilisée dans le domaine de l'analyse des signaux EEG (électroencéphalographiques) et MEG (magnétoencéphalographiques). Son objectif principal est de transformer les signaux EEG en nouvelles représentations dans lesquelles les caractéristiques discriminantes liées à des classes spécifiques sont mises en évidence et amplifiées, tandis que les caractéristiques non discriminantes sont réduites.

**Objectif:**

L'objectif du CSP est de trouver une transformation linéaire qui projette les données EEG dans un nouvel espace où les signaux sont séparés selon les classes d'intérêt. La CSP va chercher les combinaisons linéaires de canaux qui maximisent la variance entre deux classes tout en minimisant la variance au sein de chaque classe.

Le processus de CSP consiste à calculer **les matrices de covariance(1)** pour chaque classe et à les combiner pour obtenir une matrice de covariance(1) **globale**.

Ensuite, il utilise une **décomposition en valeurs propres pour trouver les vecteurs(2)** propres correspondant aux valeurs propres les plus élevées de cette matrice de covariance(1) globale. Ces vecteurs propres (appelés "filtres CSP") sont utilisés pour projeter les signaux EEG d'origine dans un nouvel espace.

**Filtres CSP:**

Les filtres CSP trouvés sont appliqués aux données EEG pour créer de nouvelles séries temporelles appelées **"composantes CSP"**.

Ces composantes sont ordonnées par ordre décroissant de variance entre les classes. Les premières composantes sont censées contenir les informations **les plus discriminantes** entre les classes, tandis que les dernières contiennent des informations moins discriminantes.

**Composantes:**

Les composantes CSP peuvent être utilisées comme caractéristiques pour des tâches de classification. En sélectionnant les composantes les plus discriminantes, on peut améliorer la séparation entre les classes et augmenter les performances de la classification.

Exemple:
On analyze des signaux EEG pour distinguer entre des états de repos et des états de mouvement. Le CSP cherchera à identifier les combinaisons de canaux EEG qui sont les plus informatives pour différencier ces deux états.

En résumé, le CSP est une technique puissante pour extraire des caractéristiques EEG discriminantes en se concentrant sur les différences inter-classes tout en atténuant les différences intra-classes. Cela permet d'améliorer la capacité des modèles de machine learning à discerner les motifs d'intérêt dans les signaux EEG.



#### **(1) Covariance**

La covariance est la moyenne des produits moins le produit des moyennes. La covariance se compare au produit des écarts-types par l'inégalité de Cauchy-Schwarz.
Autrement dit, **la covariance entre deux variables aléatoires est un nombre permettant de quantifier leurs écarts conjoints par rapport à leurs espérances respectives**.

Elle s’utilise également pour deux séries de données numériques (écarts par rapport aux moyennes). La covariance de deux variables aléatoires indépendantes est nulle, bien que la réciproque ne soit pas toujours vraie.

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/15/GaussianScatterPCA.png" width="300px">
</div>

### **(2) Décomposition en valeurs propres pour trouver les vecteurs***

**Vecteur propre**:

Le concept de vecteur propre correspond à l'étude des axes privilégiés, selon lesquels l'application se comporte comme une dilatation, multipliant les vecteurs par une même constante.

Un vecteur propre x est un vecteur non nul qui, lorsqu'il est multiplié par une matrice A, donne un résultat qui est simplement une mise à l'échelle du vecteur v par un scalaire λ.

CAD Ax = λx, où λ est la valeur propre associée à ce vecteur propre.

**Valeur propre**:

Valeur propre est le rapport de dilatation, les vecteurs auxquels il s'applique s'appellent vecteurs propres, réunis en un espace propre.

Une valeur propre d'une matrice carrée A est un scalaire λ pour lequel il existe un vecteur non nul x (appelé vecteur propre) tel que Ax = λx.

En d'autres termes, lorsque la matrice A agit sur le vecteur x, le résultat est une simple mise à l'échelle du vecteur x par le scalaire λ.

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Eigenvalue_equation.svg/2560px-Eigenvalue_equation.svg.png" width="300px">
</div>



La décomposition en valeurs propres est un processus utilisé pour décomposer une matrice carrée en une forme spéciale qui met en évidence ses valeurs propres et les vecteurs propres associés. Cela permet de simplifier l'analyse et la manipulation de la matrice. 

Nous dans notre cas, nous allons utiliser la décomposition en valeurs propres pour trouver les vecteurs propres correspondant aux valeurs propres **les plus élevées** de la matrice de covariance globale.




### Pipeline skitlearn avec CSP

Lorsque la CSP et le modèle de machine learning sont placer dans une pipeline, la pipeline crée un flux de traitement automatique qui combine ces étapes de manière séquentielle. Cela permet d'automatiser le processus d'extraction de caractéristiques avec CSP et l'entraînement du modèle de machine learning.

https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html
