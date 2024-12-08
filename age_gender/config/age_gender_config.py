from os import path

# definiram tip podataka koji ću trenirati ("age" ili "gender")
DATASET_TYPE = "gender"

# putanje do dataseta lica i izlazne putanje
BASE_PATH = "/project/adience"
OUTPUT_BASE = "output"
MX_OUTPUT = BASE_PATH

# putanje do slika i foldova
IMAGES_PATH = path.sep.join([BASE_PATH, "aligned"])
LABELS_PATH = path.sep.join([BASE_PATH, "folds"])

# postotak skupa za validaciju i testiranje u odnosu na skup za treniranje
# 70% treniranje / 15% validacija / 15% testiranje
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# hiperparametri: batch size, uređaju (1 GPU: Nvidia RTX 3070)
BATCH_SIZE = 128
NUM_DEVICES = 1

if DATASET_TYPE == "age":
    # labele za "age" dataset
    NUM_CLASSES = 8
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE, "age_le.cpickle"])

    # putanja do izlaznih train, validation, test .lst skupova
    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/age_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/age_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/age_test.lst"])

    # putanja do izlaznih train, validation, test .rec fileova
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_test.rec"])

    # datoteka sa srednjom vrijednosti pixela
    DATASET_MEAN = path.sep.join([OUTPUT_BASE, "age_adience_mean.json"])

elif DATASET_TYPE == "gender":
    # labele za "gender" dataset
    NUM_CLASSES = 2
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE, "gender_le.cpickle"])

    # putanja do izlaznih train, validation, test .lst skupova
    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/gender_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/gender_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/gender_test.lst"])

    # putanja do izlaznih train, validation, test .rec fileova
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_test.rec"])

    # datoteka sa srednjom vrijednosti pixela
    DATASET_MEAN = path.sep.join([OUTPUT_BASE, "gender_adience_mean.json"])

