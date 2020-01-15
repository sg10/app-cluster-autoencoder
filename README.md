Android App Clustering Autoencoder
==================================

Autoencoder neural network that is able to create low-dimensional, comparable representations of Android apps (embeddings) based on their Google Play description and their permission set. The obtained information can be useful as a starting point for app comparison and security-related analysis.

## Descriptions
- text represented via term frequency-inverse document frequency (TF-IDF), serves as input and output of the autoencoder
- evaluation of random layer configuration
- latent vector: 10-d


## Permissions
- one-hot encoding of the app permissions (defined in Manifest file), serves as input and output of the autoencoder
- evaluation of random layer configuration
- latent vector: 5-d

## Environment

Python 3.6: Keras (Tensorflow), Scikit-Learn, Numpy


## Evaluation & Examples

Trained on ~100,000 apps, calculation of the 2-d subspaces for descriptions and permissions for ~1,000
apps. The description t-SNE transformation performs better than PCA in clustering apps with similar
categories. Also evaluated the neighbors of apps and found that the t-SNE description subspace also
positions descriptions more closely together if looked at the actual use case, e.g., messaging apps and
browsers. For permission embeddings, anti-malware apps and web browsers were selected and analyzed
in regards to resulting clusters and outliers. A meaningful correlation between cluster positions and
permission-related functionality was found.

### Descriptions

*Neighbors in the 10d- to 2-d description subspace of WhatsApp Messenger via PCA (Principle Component Analysis) and t-SNE (Stochastic Neighborhood Embedding with t-Distribution)*

| Po | PCA                       | t-SNE                           |
| -- | ------------------------- | ------------------------------- |
| 1  | Little Photo              | Mercury Messenger (Free)        |
| 2  | Mobo Video Player Pro     | Yahoo Messenger Plug-in         |
| 3  | Textgram - Instagram Text | Telegram                        |
| 4  | Photo Warp                | Yahoo Messenger                 |
| 5  | Meme Generator Free       | Azar-Video Chat&Call, Messenger |



*Embeddings of descriptions visualized with PCA and t-SNE. The highlighted data points show that t-SNE puts apps that are in the same category closer to each other.*

![Descriptions Plot](https://github.com/sg10/app-cluster-autoencoder/blob/master/plots/descriptions.png "Descriptions Plot")

### Permissions

*Permission embeddings dissimilarity: Highlighting of anti-malware apps and web browsers in the dataset with t-SNE (5M+ downloads). Outliers and clusters are useful starting points for in-depth analysis.*

![Permissions Plot](https://github.com/sg10/app-cluster-autoencoder/blob/master/plots/permissions.png "Permissions Plot")

- **Anti-malware apps**, three outliers: According to the visualization, Antivirus Free, Antivirus for Android, and Dr. Web Lite have fairly different permission sets. These three apps do not ask the user to give them access to, e.g., location, call logs, SMS, or disk storage. The other anti-virus apps request between 33 and 81 single permissions, while the two outliers only request 5 to 14. Apps that aim to find malware on smartphones need access to critical, privacy-sensitive information to analyze suspicious behavior. For outliers that do not request such permissions, it stands to reason that they do not serve the purpose of identifying malware.
- Three isolated **browsers**: Opera Mini, Puffin, and Photon. Opera Mini is presented as a fast browser without unnecessary features. This claim seems to be correct in regard to permissions: the app only requests eight of them. Puffin and Photon are two browsers that serve mainly to run Adobe Flash scripts, but barely access other permission-critical resources. Thus, they are also positioned farther away from the other browsers. Two browsers stand out because of their launcher permissions, i.e., permissions for creating icons in the HTC app launcher or the Google app launcher. Apart from that, there are two larger groups. The blue cluster contains apps that are more intertwined with the operating system. These apps read and manage accounts stored directly on the device5, e.g., a Google or Facebook account managed by the OS. Popular examples for this group are Firefox and Chrome. Apps in the remaining group (purple) are also full-fledged browser apps (location access, audio recording, system settings, etc.) that do not request the system accounts. Overall, our approach performs well in identifying related permissions that serve a similar purpose, like for launchers or system accounts, and positions them logically correct within the subspace.
