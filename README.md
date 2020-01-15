Android App Clustering Autoencoder
==================================

Autoencoder neural network that is able to calcualte low-dimensional, comparable representations of Android apps based on their Google Play description and their permission set.

## Descriptions
- text represented via term frequency-inverse document frequency (TF-IDF), serves as input and output of the autoencoder
- evaluation of random layer configuration
- latent vector: 10-d


## Permissions
- one-hot encoding of the app permissions (defined in Manifest file), serves as input and output of the autoencoder
- evaluation of random layer configuration
- latent vector: 5-d


## Evaluation

Trained on ~100,000 apps, calculation of the 2-d subspaces for descriptions and permissions for ~1,000
apps. The description t-SNE transformation performs better than PCA in clustering apps with similar
categories. Also evaluated the neighbors of apps and found that the t-SNE description subspace also
positions descriptions more closely together if looked at the actual use case, e.g., messaging apps and
browsers. For permission embeddings, anti-malware apps and web browsers were selected and analyzed
in regards to resulting clusters and outliers. A meaningful correlation between cluster positions and
permission-related functionality was found.

### Descriptions

**Neighbors in the 10d- to 2-d description subspace of WhatsApp Messenger via PCA (Principle Component Analysis) and t-SNE (Stochastic Neighborhood Embedding with t-Distribution)**

| Po | PCA                       | t-SNE                           |
| -- | ------------------------- | ------------------------------- |
| 1  | Little Photo              | Mercury Messenger (Free)        |
| 2  | Mobo Video Player Pro     | Yahoo Messenger Plug-in         |
| 3  | Textgram - Instagram Text | Telegram                        |
| 4  | Photo Warp                | Yahoo Messenger                 |
| 5  | Meme Generator Free       | Azar-Video Chat&Call, Messenger |


**Embeddings of descriptions visualized with PCA and t-SNE. The highlighted data points
show that t-SNE puts apps that are in the same category closer to each other.**

![Descriptions Plot](https://github.com/sg10/app-cluster-autoencoder/plots/descriptions.png "Descriptions Plot")


**Permission embeddings dissimilarity: Highlighting of anti-malware apps and web
browsers in the dataset (5M+ downloads). Outliers and clusters are useful starting points
for in-depth analysis.**

![Permissions Plot](https://github.com/sg10/app-cluster-autoencoder/plots/descriptions.png "Permissions Plot")

