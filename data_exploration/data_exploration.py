import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD METADATA
df = pd.read_csv(r"C:\Users\annav\Documents\GitHub\MSU\ML_847\data\Data_Entry_2017.csv")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print(df.head())



# SIMPLE DATASET STATS
print(f"\nNumber of images: {len(df):,}")                                       # length = num of images 
print(f"Number of unique patients: {df['Patient ID'].nunique():,}")             # .nunique() counts num of unique vals, use patient ID to determine num of patients 



# EXPLORE DISEASES 
DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", 
            "Infiltration", "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax"]

all_labels = set()                                      # split pipe-separated labels into individual labels (multiple labels separated by | w/in dataset)

for labels in df["Finding Labels"]:                     # go through "Finding Labels" col for every img to split disease labels into individual labels 
    for label in labels.split("|"):
        all_labels.add(label)                           # add each individual label to set 

print("\nLabels found in dataset:")
for label in sorted(all_labels):                        # use sorted() to make output easier to read lol
    print(label)



# "NO FINDING" IMAGES
no_finding_count = (df["Finding Labels"] == "No Finding").sum()     # imgs where none of 14 diseases were assigned (T/F for every row, sum() counts T vals)
print(f"\nNo Finding images: {no_finding_count:,}")
print(f"Percentage: {no_finding_count/len(df) * 100:.2f}%")         # percent of dataset that "No Finding" represents



# HOW MANY DISASES EACH IMG HAS 
def get_disease_labels(label_string):                                   # function to convert "finding labels" col into list containing only actual disease labels 
    labels = label_string.split("|")                                    # split pipe-separated string into individual labels
    return [label for label in labels if label != "No Finding"]         # remove "no finding" bc not a disease (note that 0 = no finding)

df["Disease Labels"] = df["Finding Labels"].apply(get_disease_labels)                       # apply function to every row in df, create new col called "disease labels"
df["Num Diseases"] = df["Disease Labels"].apply(len)                                        # count num diseases presenting each img, create new col called "num diseases"
print(df[["Image Index", "Finding Labels", "Disease Labels", "Num Diseases"]].head(10))     # display first 10 rows 



# COUNT NUM DISEASES PER IMG 
disease_count_distribution = (df["Num Diseases"].value_counts().sort_index())   # value_counts() counts how frequent each num is, sort_index() puts in numerical order 
print(disease_count_distribution)

print("\nNumber of diseases per image:")

for n in [0, 1, 2, 3, 4]:                                                       # count imgs with 0 - 4 diseases 
    count = (df["Num Diseases"] == n).sum()
    print(f"{n} diseases: {count:,}")

five_plus = (df["Num Diseases"] >= 5).sum()                                     # separately count imgs wih 5 or more diseases
print(f"5+ diseases: {five_plus:,}")



# DISEASE CLASS DISTRIBUTION
class_distribution = []                                                         # store results in list 

for disease in DISEASES:                                                        # for ever img, check if disease is present, go through all 14 diseases 
    count = df["Disease Labels"].apply(lambda labels: disease in labels).sum()  # sum() counts how many T vals there are
    percentage = count/len(df) * 100                                            # percent of all imgs containing this disease 
    
    class_distribution.append({                                                 # store results in dict 
        "Disease": disease,
        "# Images": count,
        "% of Dataset": percentage
    })

class_distribution_df = pd.DataFrame(class_distribution)                                                            # convert list of dicts into df 
class_distribution_df = (class_distribution_df.sort_values("# Images", ascending = False).reset_index(drop = True)) # sort from most common to least common disease 
print(class_distribution_df)



# LOAD OFFICIAL TRAIN/TEST SPLITS 
def load_image_list(path):                                  # function to read img-list text file, return SET of filenames (set bc only care if img is present)
    with open(path, "r") as f:  
        return {line.strip() for line in f if line.strip()} # strip() removes whitespace/newline chars, prevents empty lines from being added 

train_val_images = load_image_list(r"C:\Users\annav\Documents\GitHub\MSU\ML_847\data\train_val_list.txt")   # combined train + val, not separate training and val split!
test_images = load_image_list(r"C:\Users\annav\Documents\GitHub\MSU\ML_847\data\test_list.txt")

print(f"\nTrain + Validation images: {len(train_val_images):,}")
print(f"Test images: {len(test_images):,}")
print(f"Total: {len(train_val_images) + len(test_images):,}")



# ASSIGN SPLIT TO EACH IMG 
def get_split(image_name):                                      # function to determine which split an img belongs to
    if image_name in train_val_images:                          # if img in train_val_list.txt
        return "train_val"
    elif image_name in test_images:                             # if img in test_list.txt
        return "test"
    else:                                                       # if img does not appear in either 
        return "unknown"

df["Official Split"] = (df["Image Index"].apply(get_split))     # apply function to every img in df

print("\n--- Images Per Official Split ---")
print(df["Official Split"].value_counts())                      # count how many imgs belong to each split



# CHECKING FOR PATIENT OVERLAP
train_val_patients = set(df.loc[df["Official Split"] == "train_val", "Patient ID"])     # unique IDs in train/val set
test_patients = set(df.loc[df["Official Split"] == "test", "Patient ID"])               # unique IDs in test set 
patient_overlap = (train_val_patients & test_patients)                                  # & calculates intersection of two sets

print("\n--- Patient Overlap Check ---")
print(f"Train/val patients: " f"{len(train_val_patients):,}")
print(f"Test patients: " f"{len(test_patients):,}")
print(f"Overlapping patients: " f"{len(patient_overlap):,}")