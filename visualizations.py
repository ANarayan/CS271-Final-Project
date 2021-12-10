%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def tsne_visualize(h, color, title, sample=False):
    # h = h.detach().cpu().numpy()

    if sample:
        random_idx = np.random.choice(h.shape[0], size=100)
        print(h.shape, color.shape)
        h = h[random_idx, :]
        color = color[random_idx]

    z = TSNE(n_components=2).fit_transform(h)
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.legend()
    plt.show()

def heatmap2d(arr: np.ndarray, title):
    fig, ax = plt.subplots(figsize=(20,5)) 
    plt.imshow(arr)
    plt.colorbar()
    plt.title(title)
    plt.show()

def multi_heatmap_attention(arr, title):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(15,15)) 
    
    sns.heatmap(arr[0], ax=ax1)
    sns.heatmap(arr[1], ax=ax2)
    sns.heatmap(arr[2], ax=ax3)
    sns.heatmap(arr[3], ax=ax4)
    sns.heatmap(arr[4], ax=ax5)
    sns.heatmap(arr[5], ax=ax6)
    ax1.title.set_text(f"{title} 1")
    ax2.title.set_text(f"{title} 2")
    ax3.title.set_text(f"{title} 3")
    ax4.title.set_text(f"{title} 4")
    ax5.title.set_text(f"{title} 5")
    ax6.title.set_text(f"{title} 6")
    plt.show()
    
def get_embedding_weights(model, i):
    layer_weights = model.layers[i].output
    layer_W = layer_weights[0]
    layer_b = layer_weights[1]
    return layer_W, layer_b