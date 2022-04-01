"""
Link to dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

"""

# Standard Imports
import os
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Tensorflow Keras Imports
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Global Variable Declarations
IMAGES_PATH = 'images/jpg/'

N_CLASSES = 102  # Number of classes
IMAGE_SIZE = 128  # Crop the image to 128 x 128

FLOWER_LABELS = [
    "pink primrose",  # 1
    "hard-leaved pocket orchid",  # 2
    "canterbury bells",  # 3
    "sweet pea",  # 4
    "english marigold",  # 5
    "tiger lily",  # 6
    "moon orchid",  # 7
    "bird of paradise",  # 8
    "monkshood",  # 9
    "globe thistle",  # 10
    "snapdragon",  # 11
    "colt's foot",  # 12
    "king protea",  # 13
    "spear thistle",  # 14
    "yellow iris",  # 15
    "globe-flower",  # 16
    "purple coneflower",  # 17
    "peruvian lily",  # 18
    "balloon flower",  # 19
    "giant white arum lily",  # 20
    "fire lily",  # 21
    "pincushion flower",  # 22
    "fritillary",  # 23
    "red ginger",  # 24
    "grape hyacinth",  # 25
    "corn poppy",  # 26
    "prince of wales feathers",  # 27
    "stemless gentian",  # 28
    "artichoke",  # 29
    "sweet william",  # 30
    "carnation",  # 31
    "garden phlox",  # 32
    "love in the mist",  # 33
    "mexican aster",  # 34
    "alpine sea holly",  # 35
    "ruby-lipped cattleya",  # 36
    "cape flower",  # 37
    "great masterwort",  # 38
    "siam tulip",  # 39
    "lenten rose",  # 40
    "barbeton daisy",  # 41
    "daffodil",  # 42
    "sword lily",  # 43
    "poinsettia",  # 44
    "bolero deep blue",  # 45
    "wallflower",  # 46
    "marigold",  # 47
    "buttercup",  # 48
    "oxeye daisy",  # 49
    "common dandelion",  # 50
    "petunia",  # 51
    "wild pansy",  # 52
    "primula",  # 53
    "sunflower",  # 54
    "pelargonium",  # 55
    "bishop of llandaff",  # 56
    "gaura",  # 57
    "geranium",  # 58
    "orange dahlia",  # 59
    "pink-yellow dahlia",  # 60
    "cautleya spicata",  # 61
    "japanese anemone",  # 62
    "black-eyes susan",  # 63
    "silverbush",  # 64
    "californian poppy",  # 65
    "osteospermum",  # 66
    "spring crocus",  # 67
    "bearded iris",  # 68
    "windflower",  # 69
    "tree poppy",  # 70
    "gazania",  # 71
    "azalea",  # 72
    "water lily",  # 73
    "rose",  # 74
    "thorn apple",  # 75
    "morning glory",  # 76
    "passion flower",  # 77
    "lotus",  # 78
    "toad lily",  # 79
    "anthurium",  # 80
    "frangipani",  # 81
    "clematis",  # 82
    "hibiscus",  # 83
    "columbine",  # 84
    "desert-rose",  # 85
    "tree mallow",  # 86
    "magnolia",  # 87
    "cyclamen",  # 88
    "watercress",  # 89
    "canna lily",  # 90
    "hippeastrum",  # 91
    "bee balm",  # 92
    "ball moss",  # 93
    "foxglove",  # 94
    "bougainvillea",  # 95
    "camellia",  # 96
    "mallow",  # 97
    "mexican petunia",  # 98
    "bromelia",  # 99
    "blanket flower",  # 100
    "trumpet creeper",  # 101
    "blackberry lily",  # 102
]

FLOWER_DICT = {k: v for k, v in zip(list(range(1, N_CLASSES + 1)), FLOWER_LABELS)}

CATEGORIES = np.sort(FLOWER_LABELS)


def setup():
    """
    Reads and labels images, then splits and returns dataset into training, testing, and validation subsets.
    """

    # Read flower labels from MATLAB file.
    mat = scipy.io.loadmat("imagelabels.mat")

    # Build lists of images and labels.
    images, labels = [], []
    image_dir = os.listdir(IMAGES_PATH)
    for file, index in zip(image_dir, range(len(image_dir))):
        labels.append(FLOWER_DICT[mat["labels"][0][index]])
        img = cv2.imread(os.path.join(IMAGES_PATH, file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(im)

    # Convert to numpy arrays.
    images_np = np.array(images)
    labels_np = np.array(labels)

    # Encode string labels to integers.
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_np)
    y = to_categorical(y, N_CLASSES)

    # Normalize image data from range [0, 1).
    X = images_np / 255

    # Define ratios of training, testing, and validation subsets.
    train_ratio = 0.8
    test_ratio = 0.1
    validation_ratio = 0.1

    # Split data into respective subsets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=test_ratio)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=10,
                                                          test_size=validation_ratio / (train_ratio + test_ratio))
    print(X_train.shape)
    print(X_test.shape)
    print(X_valid.shape)

    return X_train, X_test, y_train, y_test, X_valid, y_valid


def flower_classifier():
    """
    Creates and returns the CNN model.
    """

    # Instantiate the model.
    model = Sequential()

    # Begin with MobileNetV2 as a base model for speed; freeze everything but last 8 layers.
    mobile_net_v2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    for layer in mobile_net_v2.layers[:-8]:
        layer.trainable = False
    model.add(mobile_net_v2)

    # Add the rest of the model using 4 hidden Dense layers.
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # Finally, add the output layer.
    model.add(Dense(N_CLASSES, activation='softmax'))  # final layer with softmax activation

    # Display the model.
    model.summary()
    return model

def train_model(model, model_name, X_train, y_train, X_valid, y_valid, batch_size=64, epochs=256):
    """

    """

    # Create datagen object to generate batches of tensor image data.
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True)

    # Fit data generator to training data.
    datagen.fit(X_train)

    # Compile, fit, and save the model to the training data (also using the validation data).
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(datagen.flow(X_train, y_train,
                        batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_valid, y_valid),
                        verbose=1)
    model.save(model_name)

    # Gather history parameters.
    history_dict = {
        "accuracy": history.history["accuracy"],
        "validation_accuracy": history.history['val_accuracy'],
        "loss": history.history['loss'],
        "validation_loss": history.history['val_loss'],
    }

    # Save to a CSV in the model directory.
    if os.path.exists(f"{model_name}/history.csv"):
        df = pd.read_csv(f"{model_name}/history.csv")
        history_df = pd.DataFrame(data=history_dict)
        df = pd.concat([df, history_df], axis=0)
    else:
        df = pd.DataFrame(data=history_dict)
    df.to_csv(f"{model_name}/history.csv")

def plot_history(filename, model_name):
    """

    """

    # Read file.
    df = pd.read_csv(filename)
    epochs_count = len(df["accuracy"])
    epochs_np = np.array(list(range(1, epochs_count + 1)))

    # Plot training vs. validation accuracy.
    print(df["accuracy"].shape)
    plt.plot(epochs_np, df["accuracy"], 'b', label='Training Accuracy')
    plt.plot(epochs_np, df["validation_accuracy"], 'r', label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f"{model_name}/accuracy")
    plt.clf()

    # Plot training vs. validation loss.
    plt.plot(epochs_np, df["loss"], 'b', label='Training Loss')
    plt.plot(epochs_np, df["validation_loss"], 'r', label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Categorical Crossentropy Loss")
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{model_name}/loss")

def predictor(model, img_path, top_results=3):
    """

    """

    # Read the given image and normalize it.
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    doc = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    doc = np.expand_dims(doc, axis=0)

    # Make a prediction of the flower's probable categories.
    prediction = model.predict(doc)[0]
    probabilities = sorted(prediction, reverse=True)[:top_results]
    flower_indices = prediction.argsort()[-top_results:][::-1]
    for i in range(top_results):
        print(f"This flower looks like a {CATEGORIES[flower_indices[i]]} with probability {probabilities[i]:.2f}.")


def test_model(model, X_test, y_test):
    """

    """

    # Find true and predicted flower categories.
    y_true = np.argmax(y_test, axis=1)
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    # Generate and print confusion matrix and classification report.
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    print(f"Found {len(X_test)} images belonging to {len(cm)} classes")
    print('Confusion Matrix\n', cm)
    print('Classification Report\n', cr)


def random_test(model, model_name, X_test, y_test, runs=5):
    """

    """

    # Prepare to plot image predictions.
    fig, ax = plt.subplots(6, 6, figsize=(25, 40))

    test_index = 0
    for run in range(1, runs + 1):
        for i in range(6):
            for j in range(6):
                # k = int(np.random.random_sample() * len(X_test))
                truth_value = CATEGORIES[np.argmax(y_test[test_index])]
                predicted_value = CATEGORIES[np.argmax(model.predict(X_test)[test_index])]

                print(f"Iteration {i * 6 + j + 1} / 36; Testing sample {test_index}")

                # If the prediction is correct, highlight the image text green.
                if truth_value == predicted_value:
                    color = "green"
                # Otherwise, the prediction is wrong, so highlight the image text red.
                else:
                    color = "red"

                ax[i, j].set_title("TRUE: " + truth_value, color=color)
                ax[i, j].set_xlabel("PREDICTED: " + predicted_value, color=color)
                ax[i, j].imshow(np.array(X_test)[test_index].reshape(IMAGE_SIZE, IMAGE_SIZE, 3), cmap='gray')
                test_index += 1

        if not os.path.exists(f"{model_name}/runs"):
            os.mkdir(f"{model_name}/runs")
        fig.savefig(f"{model_name}/runs/0{run}.png")


def main():
    """
    Total Flower Images: 8189
    102 categories
    Train: 6551 (80%)
    Test: 819 (10%)
    Validation: 819 (10%)
    Image Size: 128 x 128 px
    Batch Size: 128
    Epochs: 300
    Epoch Duration: 13 secs
    Model Duration: 65 mins
    Total params: 24,962,470
    Trainable params: 23,435,046
    Non-trainable params: 1,527,424
    """

    model_name = "flower_prediction"
    X_train, X_test, y_train, y_test, X_valid, y_valid = setup()
    # model = flower_classifier()
    model = load_model(model_name)
    # plot_history(f"{model_name}/history.csv", model_name)
    # train_model(model, model_name, X_train, y_train, X_valid, y_valid, batch_size=128, epochs=44)
    random_test(model, model_name, X_test, y_test, runs=10)
    # predictor(model, "test_rose.jpg")
    # test_model(model, X_test, y_test)

if __name__ == "__main__": main()