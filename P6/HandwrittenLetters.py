# In[20]:

# from IPython.display import IFrame

# IFrame('https://mkpro118.github.io/AI%20Demo/HandwrittenLettersClassifier/',
#        width=1000, height=600)


# ## Setup

# ### Import Necessary Standard Libraries

# In[1]:


# Customize display of non-standard values in notebook output
from IPython.display import display, clear_output

from matplotlib import pyplot as plt

import numpy as np

from string import ascii_uppercase
import collections
import itertools
import os  # To save models

# ### Import Neural Network Library
# <a href="https://github.com/mkpro118/neural_network" target="_blank">[Library Source]</a>

# In[2]:


# Symbols are listed in order of appearance
from neural_network.layers import Dense

from neural_network.models import Sequential

from neural_network.model_selection import train_test_split
from neural_network.model_selection import StratifiedKFold

from neural_network.preprocess import Scaler
from neural_network.preprocess import OneHotEncoder

from neural_network.metrics import confusion_matrix
from neural_network.metrics import accuracy_score


# Plot interactive graphs as notebook outputs
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('dark_background')

# ---
# ## Analyzing the data

# ### Overview of the data

# In[3]:


# Read the data as int for visualization
data = np.genfromtxt('Data/handwritten_letters.csv',
                     delimiter=',', skip_header=True, dtype=int)
data.shape


# In[4]:


# Extract all columns except the first as features
features = data[:, 1:]
# Extract the first column as labels
labels = data[:, 0]
# String representation of the labels
names = ascii_uppercase

print(f'{features.shape = } | {labels.shape = }')


# ### Visualizing Handwritten Letters
#
# To get a visual understanding of the data, we can randomly select and display a handwritten letter from each class.
# <a id="visualize"></a>

# In[5]:


fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(12, 13), squeeze=False)

# plt.subplots returns an array of shape (7, 4)
# We convert that to a flat array for straightforward indexing
axs = np.ravel(axs)

# We only need the first 26 axes
for ax in axs[-2:]:
    ax.set_visible(False)
axs = axs[:-2]

rng = np.random.default_rng()

# Plot a random sample from each label
for i, label in enumerate(np.unique(labels)):
    axs[i].imshow(rng.choice(features[labels == label]
                             ).reshape(28, 28), cmap='Blues')
    axs[i].axis('off')

plt.tight_layout()
plt.show()


# ---
# ## Classifying Handwritten Letters into Uppercase English Alphabet
#
# In[6]:


def get_model() -> Sequential:
    '''Returns an untrained compiled Sequential model'''
    # Define the network
    network = [
        Dense(512, input_shape=784, activation='tanh', name='Hidden 1'),
        Dense(128, activation='tanh', name='Hidden 2'),
        Dense(26, activation='softmax', name='Output'),
    ]

    # Encapsulate the network in a Sequential model
    model = Sequential(layers=network,
                       name='Handwritten Letter Classifier')

    # Compile and build the model
    model.compile(cost='crossentropy', metrics=['accuracy_score'])

    return model


# Example Model
get_model().summary()


# ---
# ### Model Selection
#
# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    features.astype(float), labels, test_size=0.3)

print(
    f'{X_train.shape = }',
    f'{X_test.shape = }',
    f'{y_train.shape = }',
    f'{y_test.shape = }',
    sep='\n'
)


# #### 2. Stratified K-Fold Cross Validation
#
# In[8]:


n_splits = 5  # This variable is also used in a future cell
skf = StratifiedKFold(n_splits=5)

cv_splitter = skf.split(X_train, y_train)  # stratified KFold splitter


# ---
# ### Data Preprocssing
# #### 1. Rescaling
#
# In[9]:


scaler = Scaler(start=0., end=1.).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

display(f'min = {np.min(X_train)} | max = {np.max(X_train)}')


# #### 2. One-Hot Encoding the Labels
#
# In[10]:


encoder = OneHotEncoder()
encoder = encoder.fit(labels)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
display(f'{y_train.shape = } | {y_test.shape = }')


# ### Setup for Training the Neural Network Models
# #### 1. Model Initialization
#
# In[11]:


models = [get_model() for _ in range(n_splits)]


# #### 2. Initialize Trainers for Each Stratified K-Fold Split
#
# In[12]:


def get_trainers(epochs: int = 300) -> list:
    '''
    Returns a list of trainers that will run
    exactly `epochs` training iterations
    '''
    return [
        models[i].fit(
            X_train[train],
            y_train[train],
            epochs=epochs,
            steps_per_epoch=10,
            validation_data=(X_train[validate], y_train[validate]),
            verbose=False,    # The output is huge otherwise
            get_trainer=True,  # Get trainer (a generator) instead of training
        ) for i, (train, validate) in enumerate(cv_splitter)
    ]


# #### 3. Defining the Subplots for Real-Time Performance Visualization:
#
# In[13]:


def get_graphs() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    '''Returns a 2x2 grid of training tracking template graphs'''
    fig: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    ax4: plt.Axes

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    ax1.set_title('Validation Accuracy')
    ax2.set_title('Validation Loss')
    ax3.set_title('Overall Accuracy')
    ax4.set_title('Overall Loss')

    for ax in (ax1, ax3):
        ax.set_ylabel('Accuracy')
    for ax in (ax2, ax4):
        ax.set_ylabel('Loss')
    for ax in (ax1, ax3, ax3, ax4):
        ax.set_xlabel('Epochs')

    return (fig, (ax1, ax2, ax3, ax4))


# Number of training iterations between graph updates
update_frequency = 2

# Each color represents a model on the graphs
colors = ['lightblue', 'hotpink', 'lightgreen', 'orange', 'red']


# #### 4. Training History and Best Metrics Tracking:
#
# In[14]:


def get_history_dict() -> dict:
    history: dict = {
        num: {
            'overall': collections.defaultdict(list),
            'validation': collections.defaultdict(list),
        } for num in range(len(models))
    }

    history.update({
        'best': {
            'overall': {
                'accuracy': -np.inf,
                'loss': np.inf,
            },
            'validation': {
                'accuracy': -np.inf,
                'loss': np.inf,
            },
        },
    })
    return history


# ### Training the Models
#
# In[15]:


def train(epochs: int = 300):
    '''Runs the training loop'''
    trainers = get_trainers(epochs)
    history = get_history_dict()

    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    ax4: plt.Axes

    fig, (ax1, ax2, ax3, ax4) = get_graphs()  # type: ignore[misc]
    clear_output(wait=False)

    for iterations in itertools.count(start=1):
        try:
            # Run a training iteration
            data: tuple = tuple(map(next, trainers))
            # Here we use a list comprehension instead of
            # normal for loops because list comprehensions
            # are slightly faster
            [
                history[num][type_][metric].append(data[num][type_][metric])
                for num in range(len(models))
                for type_ in data[num].keys()
                for metric in data[num][type_].keys()
            ]
        except (StopIteration, IndexError):  # Generators stop as training completes
            break

        # Lazy updates to speed up training
        if iterations % update_frequency != 0:
            continue

        # Update with iterations count
        fig.suptitle(f'After {iterations} Iterations', fontsize=24, y=1.04)

        # Clear all axes to plot again
        for ax in (ax1, ax2, ax3, ax4):
            ax.cla()

        # Live Plots
        for num in range(len(models)):
            # Update Validation Accuray Plot
            ax1.plot(
                np.arange(len(history[num]['validation']['accuracy'])) + 1,
                history[num]['validation']['accuracy'],
                color=colors[num],
                label=f'Model {num + 1}'
            )

            # Update Validation Loss Plot
            ax2.plot(
                np.arange(len(history[num]['validation']['loss'])) + 1,
                history[num]['validation']['loss'],
                color=colors[num],
                label=f'Model {num + 1}'
            )

            # Update Overall Accuray Plot
            ax3.plot(
                np.arange(len(history[num]['overall']['accuracy'])) + 1,
                history[num]['overall']['accuracy'],
                color=colors[num],
                label=f'Model {num + 1}'
            )

            # Update Overall Loss Plot
            ax4.plot(
                np.arange(len(history[num]['overall']['loss'])) + 1,
                history[num]['overall']['loss'],
                color=colors[num],
                label=f'Model {num + 1}'
            )

        # Update the titles with best accuracy and loss information
        x = max(history[num]["validation"]["accuracy"][-1]
                for num in range(len(models)))
        x = max(history['best']['validation']['accuracy'], x)
        history['best']['validation']['accuracy'] = x
        ax1.set_title(
            f'Best Validation Accuracy: {np.around(x * 100, 2)}%', fontsize=20, pad=20)
        ax1.legend(loc='upper left')

        x = min(history[num]["validation"]["loss"][-1]
                for num in range(len(models)))
        x = min(history['best']['validation']['loss'], x)
        history['best']['validation']['loss'] = x
        ax2.set_title(
            f'Best Validation Loss: {np.around(x, 4)}', fontsize=20, pad=20)
        ax2.legend(loc='upper right')

        x = max(history[num]["overall"]["accuracy"][-1]
                for num in range(len(models)))
        x = max(history['best']['overall']['accuracy'], x)
        history['best']['overall']['accuracy'] = x
        ax3.set_title(
            f'Best Overall Accuracy: {np.around(x * 100, 2)}%', fontsize=20, pad=20)
        ax3.legend(loc='upper left')

        x = min(history[num]["overall"]["loss"][-1]
                for num in range(len(models)))
        x = min(history['best']['overall']['loss'], x)
        history['best']['overall']['loss'] = x
        ax4.set_title(
            f'Best Overall Loss: {np.around(x, 4)}', fontsize=20, pad=20)
        ax4.legend(loc='upper right')

        for ax in (ax2, ax4):
            ax.set_xlabel('Iterations', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)

        for ax in (ax1, ax3):
            ax.set_xlabel('Iterations', fontsize=18)
            ax.set_ylabel('Accuracy', fontsize=18)

        for ax in (ax1, ax2, ax3, ax4):
            ax.tick_params(axis='both', labelsize=18)

        fig.tight_layout()

        clear_output(wait=False)
        display(fig)

        # This is required to completely draw the plots
        # See https://stackoverflow.com/a/53760972
        plt.pause(0.1)


train()


# ### Saving the model
#
# In[16]:


def save_models():
    if not os.path.isdir('Saved Models'):
        os.makedirs('Saved Models')
    for i, model in enumerate(models, start=1):
        model.save(f'Saved Models/Model {i}.json')


save_models()


# ### Testing the Model
# #### Making the Predictions
#
# In[17]:


predictions = [model.predict(X_test, classify=True) for model in models]


# #### Model Accuracy
#
# In[18]:


model_names = [f'Model {i + 1}' for i in range(len(predictions))]

accuracies = [accuracy_score(y_true=y_test, y_pred=pred)
              * 100 for pred in predictions]
accuracies = np.around(accuracies, 2)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
ax.bar(model_names, accuracies, color='lightblue')
ax.set_xlabel('Models', fontsize=16)
ax.set_ylabel('Accuracy (%)', fontsize=16)
ax.set_title('Model Accuracies on Holdout Data', fontsize=20, pad=20)
ax.set_xticks(np.arange(5), labels=model_names, fontsize=12, rotation=45)
ax.set_yticks(np.arange(0, 110, 20))

for i, acc in enumerate(accuracies):
    plt.text(x=i, y=acc + 1, s=f'{acc}%', ha='center', fontsize=12)

plt.tight_layout()
plt.show()


# ### Visualizing Misclassifications with Confusion Matrix
#
# In[19]:


cmats = [confusion_matrix(y_test, prediction) for prediction in predictions]

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 25))
axs = np.ravel(axs)
axs[-1].axis('off')
axs = axs[:-1]

for i, (ax, cmat) in enumerate(zip(axs, cmats), start=1):
    ax.matshow(cmat, cmap='Blues', alpha=0.6)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(map(str, names), fontsize=14)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(map(str, names), fontsize=14)
    ax.set_title(f'Model {i}', fontsize=18, pad=20)

    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(x=j, y=i, s=cmat[i, j], va='center',
                    ha='center', size='small', color='white')
plt.show()
