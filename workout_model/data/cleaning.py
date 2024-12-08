import pandas as pd
import os
from workout_model.common.paths import paths
from workout_model.util.utilities import fileStandardSortKey, cleanBadCharacters



#region === Initial Cleaning ===

# This function imports the raw gym exercise data into a pandas dataframe.
def readGymExerciseCSV():
    datasetPath = paths["data"] / "gym_exercise_dataset.csv"
    _rawGymExerciseDataFrame = pd.read_csv(datasetPath)
    return _rawGymExerciseDataFrame

# This function simply drops the unneeded columns from the dataframe, and divides multi-entry cells into lists.
def prepareColumns(_gymExerciseDataFrame):
    # Remove unneeded columns.
    _gymExerciseDataFrame = _gymExerciseDataFrame.drop(columns=["Preparation","Execution","Force","Secondary Muscles",
                                                                "Stabilizer_Muscles","Antagonist_Muscles",
                                                                "Dynamic_Stabilizer_Muscles","parent_id","Difficulty (1-5)"])

    # Standardize string values in all columns
    #TODO does this truncate a single value for some reason?
    for column in _gymExerciseDataFrame.columns:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(
            lambda x: ' '.join(str(x).split()) if pd.notnull(x) else x
        )

    # Clean up bad/invisible characters from all values.
    for column in _gymExerciseDataFrame.columns:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(cleanBadCharacters)

    return _gymExerciseDataFrame

# This function standardizes all values within the dataframe that are given in variations to standard. For example,
# if standard is "On", and variations is ["Active", "working", "on", "Not Off"], all values found equivalent to
# variations will be converted to "On".
def standardizeValue(_gymExerciseDataFrame,standard : str,variations : (str,list), target_columns: list = None,
                     lookupColumn : (str,list) = None):
    # Ensure variations is a list for consistent processing
    if isinstance(variations, str):
        variations = [variations]

    # Ensure lookupColumn is a list
    if isinstance(lookupColumn, str):
        lookupColumn = [lookupColumn]

    # Normalize case and whitespace for consistency in matching
    variations = [v.strip().lower() for v in variations]
    standard = standard.strip()

    # Helper function to replace variations with the standard
    def replace_variations(value):
        if isinstance(value, str) and value.strip().lower() in variations:
            return standard
        return value

    # Standardize values in the target_column(s)
    target_column = target_columns[0]
    _gymExerciseDataFrame[target_column] = _gymExerciseDataFrame[target_column].apply(replace_variations)

    # If lookupColumn is provided, handle deduplication
    if lookupColumn:
        # Group by lookup columns
        def deduplicate_group(group):
            # If there's more than one row in the group, keep one and update its target_column to the standard
            if len(group) > 1:
                group = group.iloc[:1]  # Keep only the first row
                group[target_column] = standard
            return group

        # Group by lookup columns and apply deduplication logic
        _gymExerciseDataFrame = _gymExerciseDataFrame.groupby(lookupColumn, group_keys=False).apply(deduplicate_group)

    return _gymExerciseDataFrame

# This method deduplicates tuples based on the uniqueness of lookupColumn(s) with a subset of rows, defined by
# variations in the target_columns.
def deduplicateByLookup(_gymExerciseDataFrame, variations: (str, list), targetColumn: str, lookupColumns: (str, list)):
    # Ensure variations is a list
    if isinstance(variations, str):
        variations = [variations]

    # Ensure lookupColumns is a list
    if isinstance(lookupColumns, str):
        lookupColumns = [lookupColumns]

    # Normalize variations for consistent matching
    variations = [v.strip().lower() for v in variations]

    # Step 1: Filter rows based on variations in the target_column
    filtered_subset = _gymExerciseDataFrame[_gymExerciseDataFrame[targetColumn]
                                            .str.strip()
                                            .str.lower()
                                            .isin(variations)]

    # Step 2: Deduplicate the filtered subset based on lookupColumns uniqueness
    # Sort to prioritize a specific variation (optional, can sort by other criteria)
    filtered_subset = filtered_subset.sort_values(by=targetColumn)

    # Drop duplicates, keeping only the first occurrence based on lookupColumns
    deduplicated_subset = filtered_subset.drop_duplicates(subset=lookupColumns)

    # Step 3: Merge the deduplicated subset with the rest of the original dataframe
    remaining_data = _gymExerciseDataFrame[~_gymExerciseDataFrame.index.isin(filtered_subset.index)]
    result_df = pd.concat([remaining_data, deduplicated_subset])

    return result_df

#endregion === Initial Cleaning ===
#region === Data Removal ===

# This function removes all tuples that contain equipments we don't want to worry about presenting to the user
# (assisted, plyometric)
def removeBadEquipment(_gymExerciseDataFrame):
    badEquipmentTypes = ["Assisted","Assisted (Partner)","Assisted Chest Dip","Plyometric"]
    return _gymExerciseDataFrame[~_gymExerciseDataFrame["Equipment"].isin(badEquipmentTypes)]

#endregion === Data Removal ===
#region === Data Preprocessing ===

# Some muscles were inconsistently stored with commas in them - this dict corrects it.
def applyMuscleCorrections(muscle_string):
    muscle_corrections = {"Trapezius, Upper": "Upper Trapezius",
                          "Trapezius, Middle": "Middle Trapezius",
                          "Trapezius, Lower": "Lower Trapezius",
                          "Pectoralis Major, Sternal": "Sternal Pectoralis Major",
                          "Erector Spinae, Cervicis & Capitis Fibers": "Cervicis & Capitis Erector Spinae",
                          "Sternocleidomastoid, Posterior Fibers": "Posterior Fibers Sternocleidomastoid",}

    if pd.isnull(muscle_string):
        # Return as-is if the value is NaN
        return muscle_string

    # Replace problematic substrings using the correction dictionary
    for incorrect, correct in muscle_corrections.items():
        muscle_string = muscle_string.replace(incorrect, correct)

    return muscle_string

# This function uses multi-hot encoding to represent the list of all muscles as features, rather than vague lists,
# then removes the original muscles features.
# Multi-hot encodes muscles with distinction:
#     1 = Target muscle
#     2 = Synergist muscle
#     0 = None
def multiHotEncodeMuscles(_gymExerciseDataFrame):
    # First, we apply some corrections due to inconsistent comma inclusion in the dataset
    columnsToProcess = ["Target_Muscles", "Synergist_Muscles"]
    for column in columnsToProcess:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(
            lambda x: applyMuscleCorrections(x)
        )

    # Now we process our muscle columns, which may contain, multiple values into lists.
    for column in columnsToProcess:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(
            lambda x: [item.strip() for item in str(x).split(",") if item.strip()] if pd.notnull(x) else []
        )

    # Generate a set of all unique muscles values in ONLY Target_Muscles.
    uniqueMuscles = set()
    for idx, row in _gymExerciseDataFrame.iterrows():
        targetMuscles = row["Target_Muscles"]
        for targetMuscle in targetMuscles:
            uniqueMuscles.add(targetMuscle)

    # Initialize new columns for each unique muscle
    for muscle in uniqueMuscles:
        _gymExerciseDataFrame[muscle] = 0.0  # Default to 0 for all rows

    # Update each muscle column based on primary and secondary muscles
    for idx, row in _gymExerciseDataFrame.iterrows():
        targetMuscles = row["Target_Muscles"]
        synergistMuscles = row["Synergist_Muscles"]

        for muscle in uniqueMuscles:
            if muscle in targetMuscles:
                _gymExerciseDataFrame.at[idx, muscle] = 1.0  # Primary muscle
            elif muscle in synergistMuscles:
                _gymExerciseDataFrame.at[idx, muscle] = 0.5  # Secondary muscle

    # Drop the original muscle columns
    _gymExerciseDataFrame = _gymExerciseDataFrame.drop(columns=["Target_Muscles", "Synergist_Muscles"])

    return _gymExerciseDataFrame

#endregion === Data Preprocessing ===
#region === Reports ===

# Reports are used exclusively

# This function identifies unique values across all columns, including list columns, to detect unique values and
# inconsistencies manually.
def generateUniqueValueReport(_gymExerciseDataFrame,blacklistColumns = None):
    if(not blacklistColumns):
        blacklistColumns = []

    # Make path if missing
    uniquenessReportsPath = paths["reports"] / "uniqueness_reports"
    if(not os.path.exists(uniquenessReportsPath)):
        os.mkdir(uniquenessReportsPath)

    # Loop through and report on all columns
    for column in _gymExerciseDataFrame.columns:
        if(column in blacklistColumns):
            continue
        # Temporarily convert list-like columns to strings for processing
        tempColumn = _gymExerciseDataFrame[column].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        # Generate the sorted list of unique values
        uniqueValues = sorted(tempColumn.unique(), key=fileStandardSortKey)

        # Print the column name and unique values
        reportString = f"{column}\n\n"
        for uniqueValue in uniqueValues:
            reportString += f"{uniqueValue}\n"
        with(open(uniquenessReportsPath / f"{column}.txt","w",encoding="utf-8") as f):
            f.write(reportString)

#endregion === Reports ===



#region === Process ===

uniqueMuscles = ["Anterior Deltoid",
"Biceps Brachii",
"Brachialis",
"Brachioradialis",
"Gastrocnemius",
"Gluteus Maximus",
"Hamstrings",
"Hip Abductors",
"Hip Adductors",
"Hip Flexors",
"Infraspinatus",
"Lateral Deltoid",
"Latissimus Dorsi",
"Levator Scapulae",
"Lower Trapezius",
"Middle Trapezius",
"Pectoralis Major Clavicular",
"Pectoralis Major Sternal",
"Posterior Deltoid",
"Pronators",
"Quadriceps",
"Rhomboids",
"Serratus Anterior",
"Soleus",
"Splenius",
"Sternocleidomastoid",
"Subscapularis",
"Supinator",
"Supraspinatus",
"Teres Major",
"Teres Minor",
"Tibialis Anterior",
"Triceps Brachii",
"Upper Trapezius",
"Wrist Extensors",
"Wrist Flexors"]

gymExerciseDataFrame = readGymExerciseCSV()
gymExerciseDataFrame = prepareColumns(gymExerciseDataFrame)

gymExerciseDataFrame = removeBadEquipment(gymExerciseDataFrame)
gymExerciseDataFrame = multiHotEncodeMuscles(gymExerciseDataFrame)


# Here, we use deduplication to remove (as defined by our research) distinction-less exercise tuples from the data.
# We reference "grouped_exercises.txt", which is a document we prepared using generated uniqueness reports to group
# reasonably similar exercises together, and then let our algorithm detect which tuples are worth preserving and
# which can be dropped.
groupedExercises = []
with open(paths["data"] / "grouped_exercises.txt","r") as f:
    for line in f:
        groupedExercises.append([thisExercise.strip() for thisExercise in line.split(",")])
for groupedExercise in groupedExercises:
    gymExerciseDataFrame = deduplicateByLookup(gymExerciseDataFrame,variations=groupedExercise,
                                               targetColumn="Exercise Name",lookupColumns=["Equipment"] + uniqueMuscles)


generateUniqueValueReport(gymExerciseDataFrame,blacklistColumns=uniqueMuscles)
print(gymExerciseDataFrame.head(20).to_string())
#print(gymExerciseDataFrame[gymExerciseDataFrame["Exercise Name"].isin(["Neck Flexion","Lateral Neck Flexion"])].to_string())

#endregion === Process ===